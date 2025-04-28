from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import os
import uuid
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
import logging  # Add this import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Define the logger

# Load environment variables
load_dotenv()

# Initialize API
app = FastAPI(title="Diagram RAG Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings and configurations
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")

# Security
security = HTTPBearer()

# Set up Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Supabase client
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Models
class UserSignUp(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserSignIn(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    diagram_path: Optional[str] = None
    session_id: str

class ChatSession(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserProfile

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days default
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Generate embeddings for vector search
def generate_embedding(text: str) -> List[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# Search diagrams function
def search_diagrams(query: str, top_k: int = 1):
    try:
        supabase = get_supabase()
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Fetch all diagram embeddings
        response = supabase.table("diagram_embeddings").select(
            "diagram_id, embedding"
        ).execute()
        
        # Manual vector similarity calculation
        results = []
        for item in response.data:
            try:
                # Get the embedding data
                embedding_data = item['embedding']
                
                # Check if embedding is a string and convert it to a list if needed
                if isinstance(embedding_data, str):
                    # If it starts with '[' and ends with ']', it's likely a string representation of a list
                    if embedding_data.startswith('[') and embedding_data.endswith(']'):
                        # Convert string representation to actual list of floats
                        import ast
                        embedding_data = ast.literal_eval(embedding_data)
                    else:
                        # Skip this item if the embedding format is unexpected
                        logger.warning(f"Unexpected embedding format for diagram {item.get('diagram_id')}")
                        continue
                
                # Convert embeddings to numpy arrays of float type
                db_embedding = np.array(embedding_data, dtype=np.float32)
                query_emb = np.array(query_embedding, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(db_embedding, query_emb) / (np.linalg.norm(db_embedding) * np.linalg.norm(query_emb))
                
                # Get diagram details
                diagram_response = supabase.table("diagrams").select(
                    "id, filename, description, source_pdf, page_number, storage_path"
                ).eq("id", item['diagram_id']).execute()
                
                if diagram_response.data:
                    diagram = diagram_response.data[0]
                    
                    # Get storage URL
                    bucket_url = f"{SUPABASE_URL}/storage/v1/object/public/diagrams"
                    diagram_path = f"{bucket_url}/{diagram['storage_path']}"
                    
                    results.append({
                        'id': diagram['id'],
                        'diagram_path': diagram_path,
                        'description': diagram.get('description', ''),
                        'source_pdf': diagram.get('source_pdf', ''),
                        'page_number': diagram.get('page_number', ''),
                        'similarity': float(similarity)
                    })
            except Exception as e:
                logger.error(f"Error processing embedding for diagram {item.get('diagram_id')}: {e}")
                continue
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        logger.error(f"Error searching diagrams: {e}")
        return []

# User Authentication Endpoints
@app.post("/auth/signup", response_model=TokenResponse)
async def signup(user_data: UserSignUp):
    try:
        supabase = get_supabase()
        
        # Create user in Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": user_data.email,
            "password": user_data.password
        })
        
        if auth_response.user:
            # Create user profile in the profiles table
            user_id = auth_response.user.id
            
            # Insert into profiles table
            profile_data = {
                "id": user_id,
                "email": user_data.email,
                "full_name": user_data.full_name
            }
            
            profile_response = supabase.table("profiles").insert(profile_data).execute()
            
            # Create access token
            access_token = create_access_token({"sub": user_id})
            
            # Format user profile for response
            user_profile = UserProfile(
                id=user_id,
                email=user_data.email,
                full_name=user_data.full_name,
                created_at=datetime.utcnow()
            )
            
            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                user=user_profile
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create user")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during signup: {str(e)}")

@app.post("/auth/signin", response_model=TokenResponse)
async def signin(user_data: UserSignIn):
    # Log the sign-in attempt
    logger.info(f"Sign-in attempt for: {user_data.email}")
    
    try:
        # Validate Supabase configuration
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error(f"Supabase configuration missing. URL: {bool(SUPABASE_URL)}, Key: {bool(SUPABASE_KEY)}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service not properly configured"
            )
            
        # Log Supabase connection attempt
        logger.info(f"Attempting to connect to Supabase at: {SUPABASE_URL}")
        
        # Get Supabase client without custom options
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client created successfully")
        except Exception as conn_err:
            logger.error(f"Failed to create Supabase client: {str(conn_err)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to authentication service: {str(conn_err)}"
            )
        
        # Perform the authentication
        try:
            logger.info("Attempting authentication with Supabase")
            # Try the updated authentication method
            auth_response = supabase.auth.sign_in_with_password({
                "email": user_data.email,
                "password": user_data.password
            })
            
            logger.info("Authentication response received")
            
            # Check if user exists in the response
            if not hasattr(auth_response, 'user') or not auth_response.user:
                # Try alternative response structure
                if hasattr(auth_response, 'data') and hasattr(auth_response.data, 'user'):
                    user = auth_response.data.user
                elif isinstance(auth_response, dict) and 'user' in auth_response:
                    user = auth_response['user']
                else:
                    logger.warning(f"Invalid credentials for user: {user_data.email}")
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid credentials"
                    )
            else:
                user = auth_response.user
            
            user_id = user.id if hasattr(user, 'id') else user.get('id')
            logger.info(f"User authenticated successfully. User ID: {user_id}")
            
        except Exception as auth_err:
            logger.error(f"Authentication error: {str(auth_err)}")
            if "Invalid login credentials" in str(auth_err):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid email or password"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Authentication error: {str(auth_err)}"
                )
        
        # Get user profile
        try:
            logger.info(f"Fetching user profile for ID: {user_id}")
            profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
            
            if not profile_response.data or len(profile_response.data) == 0:
                logger.warning(f"User profile not found for ID: {user_id}")
                
                # Create a basic profile if not found
                now = datetime.utcnow().isoformat()
                user_profile = {
                    "id": user_id,
                    "email": user_data.email,
                    "full_name": None,
                    "created_at": now
                }
                
                logger.info(f"Creating new user profile for ID: {user_id}")
                profile_response = supabase.table("profiles").insert(user_profile).execute()
                user_profile = profile_response.data[0] if profile_response.data else user_profile
            else:
                user_profile = profile_response.data[0]
                logger.info(f"User profile retrieved successfully for ID: {user_id}")
        
        except Exception as profile_err:
            logger.error(f"Error fetching user profile: {str(profile_err)}")
            # Continue with basic profile if we can't get the full one
            user_profile = {
                "id": user_id,
                "email": user_data.email,
                "full_name": None,
                "created_at": datetime.utcnow().isoformat()
            }
        
        # Generate JWT token
        try:
            logger.info(f"Generating access token for user ID: {user_id}")
            access_token = create_access_token({"sub": user_id})
            logger.info("Access token generated successfully")
            
            # Ensure we have the necessary fields
            created_at = user_profile.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at_datetime = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at_datetime = datetime.utcnow()
            else:
                created_at_datetime = datetime.utcnow()
            
            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                user=UserProfile(
                    id=user_id,
                    email=user_profile.get("email", user_data.email),
                    full_name=user_profile.get("full_name"),
                    created_at=created_at_datetime
                )
            )
            
        except Exception as token_err:
            logger.error(f"Error generating token: {str(token_err)}")
            raise HTTPException(
                status_code=500,
                detail="Error generating authentication token"
            )
    
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        raise http_ex
    except Exception as e:
        # Log the full error with traceback
        import traceback
        logger.error(f"Unexpected error during signin: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Authentication system error: {str(e)}"
        )
    
@app.get("/auth/me", response_model=UserProfile)
async def get_current_user(payload: Dict = Depends(verify_token)):
    try:
        supabase = get_supabase()
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        # Get user profile
        profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
        
        if profile_response.data:
            profile = profile_response.data[0]
            return UserProfile(
                id=user_id,
                email=profile.get("email"),
                full_name=profile.get("full_name"),
                created_at=profile.get("created_at", datetime.utcnow())
            )
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
            
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Chat Sessions Endpoints
@app.get("/chat/sessions", response_model=List[ChatSession])
async def get_chat_sessions(payload: Dict = Depends(verify_token)):
    try:
        supabase = get_supabase()
        user_id = payload.get("sub")
        
        # Get user's chat sessions
        response = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        
        if response.data:
            # Format sessions
            sessions = []
            for session in response.data:
                sessions.append(ChatSession(
                    id=session.get("id"),
                    name=session.get("name"),
                    created_at=session.get("created_at"),
                    updated_at=session.get("updated_at")
                ))
            return sessions
        else:
            return []
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat sessions: {str(e)}")

@app.post("/chat/sessions", response_model=ChatSession)
async def create_chat_session(name: str = "New Chat", payload: Dict = Depends(verify_token)):
    try:
        supabase = get_supabase()
        user_id = payload.get("sub")
        
        now = datetime.utcnow()
        
        # Create new chat session
        session_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        response = supabase.table("chat_sessions").insert(session_data).execute()
        
        if response.data:
            session = response.data[0]
            return ChatSession(
                id=session.get("id"),
                name=session.get("name"),
                created_at=session.get("created_at"),
                updated_at=session.get("updated_at")
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create chat session")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating chat session: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, payload: Dict = Depends(verify_token)):
    try:
        supabase = get_supabase()
        user_id = payload.get("sub")
        
        # Get or create session ID
        session_id = request.session_id
        if not session_id:
            # Create new session
            now = datetime.utcnow().isoformat()
            session_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "name": "New Chat",
                "created_at": now,
                "updated_at": now
            }
            response = supabase.table("chat_sessions").insert(session_data).execute()
            if response.data:
                session_id = response.data[0].get("id")
            else:
                raise HTTPException(status_code=500, detail="Failed to create chat session")
        
        # Get chat history from database
        history_response = supabase.table("chat_messages").select(
            "role, content"
        ).eq("session_id", session_id).order("created_at", desc=False).execute()
        
        history = []
        if history_response.data:
            # Convert to format expected by Gemini
            for message in history_response.data:
                history.append({
                    "role": message.get("role"),
                    "parts": [message.get("content")]
                })
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # Determine if the query might need a diagram
        needs_diagram_prompt = f"""
        Analyze this educational query: "{request.prompt}"
        
        Does this query potentially need or reference a diagram, chart, architecture, 
        or visual representation to properly explain the concept? Answer with YES or NO only.
        """
        
        needs_diagram_response = model.generate_content(needs_diagram_prompt).text.strip()
        
        # Search for relevant diagrams if needed
        retrieved_context = ""
        diagram_path = None
        
        if "YES" in needs_diagram_response.upper():
            diagram_results = search_diagrams(request.prompt)
            
            if diagram_results and diagram_results[0]['similarity'] > 0.6:
                best_match = diagram_results[0]
                diagram_path = best_match['diagram_path']
                
                retrieved_context = f"""
                I found a relevant diagram that helps explain this concept.
                Description: {best_match['description']}
                Source: {best_match.get('source_pdf', 'Educational material')}, 
                page {best_match.get('page_number', 'N/A')}
                
                Please refer to the attached diagram while reading my explanation.
                """
        
        # Start chat with history
        chat = model.start_chat(history=history)
        
        # Add system instructions to the user message
        system_instructions = """
        You are an educational assistant that helps explain concepts clearly.
        When diagrams are available, refer to them in your explanation to enhance understanding.
        Keep explanations clear, concise, and tailored to educational purposes.
        """
        
        # Construct prompt with system instructions and retrieved diagram info
        user_message = f"""
        {system_instructions}
        
        User query: {request.prompt}
        
        {retrieved_context if retrieved_context else ""}
        """
        
        # Get response
        response = chat.send_message(user_message)
        response_text = response.text
        
        # Update chat history in database
        now = datetime.utcnow().isoformat()
        
        # Store user message
        user_message_data = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "role": "user",
            "content": request.prompt,
            "created_at": now
        }
        supabase.table("chat_messages").insert(user_message_data).execute()
        
        # Store assistant message
        assistant_message_data = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "role": "model",
            "content": response_text,
            "created_at": now,
            "diagram_path": diagram_path
        }
        supabase.table("chat_messages").insert(assistant_message_data).execute()
        
        # Update session last updated timestamp
        supabase.table("chat_sessions").update(
            {"updated_at": now, "name": request.prompt[:30] + "..." if len(request.prompt) > 30 else request.prompt}
        ).eq("id", session_id).execute()
        
        # Prepare response
        return ChatResponse(
            response=response_text,
            diagram_path=diagram_path,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, payload: Dict = Depends(verify_token)):
    try:
        supabase = get_supabase()
        user_id = payload.get("sub")
        
        # First check if session belongs to user
        session_response = supabase.table("chat_sessions").select("*").eq("id", session_id).eq("user_id", user_id).execute()
        
        if not session_response.data:
            raise HTTPException(status_code=403, detail="You don't have access to this chat session")
        
        # Get chat messages
        messages_response = supabase.table("chat_messages").select(
            "id, role, content, created_at, diagram_path"
        ).eq("session_id", session_id).order("created_at", desc=False).execute()
        
        return {"messages": messages_response.data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "online", "message": "Diagram RAG Chatbot server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
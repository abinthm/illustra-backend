# Diagram RAG Chatbot

A FastAPI-based chatbot application that combines Retrieval-Augmented Generation (RAG) with diagram generation capabilities. This application allows users to interact with an AI assistant that can generate and retrieve diagrams based on natural language queries.

## Features

- User authentication and authorization
- Chat session management
- Diagram generation and retrieval
- RAG-based responses
- Secure API endpoints
- CORS support
- JWT-based authentication

## Tech Stack

- **Backend**: FastAPI
- **Database**: Supabase
- **AI/ML**: 
  - Google Gemini AI
  - Sentence Transformers
- **Authentication**: JWT
- **Environment**: Python 3.10

## Prerequisites

- Python 3.10
- Conda (for environment management)
- Supabase account
- Google Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate the Conda environment:
```bash
conda env create -f requirements.yaml
conda activate diagram-rag-chatbot-env
```

3. Create a `.env` file in the root directory with the following variables:
```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
JWT_SECRET_KEY=your_jwt_secret_key
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
- `POST /auth/signup` - Register a new user
- `POST /auth/signin` - Login existing user
- `GET /auth/me` - Get current user profile

### Chat
- `GET /chat/sessions` - List all chat sessions
- `POST /chat/sessions` - Create a new chat session
- `POST /chat` - Send a message and get response
- `GET /chat/history/{session_id}` - Get chat history for a session

## Security

- All endpoints except `/auth/signup` and `/auth/signin` require JWT authentication
- CORS is configured to allow cross-origin requests
- Environment variables are used for sensitive configuration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

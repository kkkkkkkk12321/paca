"""
PACA Production Server
FastAPI-based production server for PACA system
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
import structlog

# PACA 시스템 imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paca.tools import ReActFramework, PACAToolManager, Tool, ToolResult
from paca.tools.tools.web_search import WebSearchTool
from paca.tools.tools.file_manager import FileManagerTool

# Metrics
REQUEST_COUNT = Counter('paca_requests_total', 'Total number of requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('paca_request_duration_seconds', 'Request duration in seconds')
TOOL_EXECUTION_COUNT = Counter('paca_tool_executions_total', 'Total tool executions', ['tool_name', 'status'])

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

class PACARequest(BaseModel):
    """PACA 요청 모델"""
    session_id: str = Field(..., description="세션 ID")
    action: str = Field(..., description="실행할 액션 (think/act/observe/reflect)")
    content: str = Field(..., description="액션 내용")
    tool_name: Optional[str] = Field(None, description="사용할 도구 이름")
    tool_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="도구 매개변수")
    confidence: Optional[float] = Field(0.5, description="신뢰도 (0.0-1.0)")

class PACAResponse(BaseModel):
    """PACA 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    session_id: str = Field(..., description="세션 ID")
    step_id: str = Field(..., description="스텝 ID")
    action: str = Field(..., description="실행된 액션")
    content: str = Field(..., description="응답 내용")
    timestamp: datetime = Field(default_factory=datetime.now, description="실행 시간")
    confidence: Optional[float] = Field(None, description="신뢰도")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="도구 실행 결과")
    error: Optional[str] = Field(None, description="오류 메시지")

class ProductionConfig:
    """프로덕션 설정"""
    def __init__(self):
        # Environment variables with defaults
        self.env = os.getenv('PACA_ENV', 'production')
        self.debug = os.getenv('PACA_DEBUG', 'false').lower() == 'true'
        self.host = os.getenv('PACA_HOST', '0.0.0.0')
        self.port = int(os.getenv('PACA_PORT', 8000))
        self.workers = int(os.getenv('PACA_WORKERS', 4))

        # Security
        self.secret_key = os.getenv('PACA_SECRET_KEY', 'default-secret-key')
        self.allowed_origins = self._parse_origins(os.getenv('PACA_ALLOW_ORIGINS', '["*"]'))

        # API limits
        self.api_timeout = int(os.getenv('PACA_API_TIMEOUT', 30))
        self.max_requests_per_minute = int(os.getenv('PACA_MAX_REQUESTS_PER_MINUTE', 100))
        self.max_tool_execution_time = int(os.getenv('PACA_MAX_TOOL_EXECUTION_TIME', 300))

        # Tool configuration
        self.enable_web_search = os.getenv('PACA_ENABLE_WEB_SEARCH', 'true').lower() == 'true'
        self.enable_file_manager = os.getenv('PACA_ENABLE_FILE_MANAGER', 'true').lower() == 'true'
        self.sandbox_mode = os.getenv('PACA_SANDBOX_MODE', 'true').lower() == 'true'

        # Monitoring
        self.metrics_enabled = os.getenv('PACA_METRICS_ENABLED', 'true').lower() == 'true'
        self.metrics_port = int(os.getenv('PACA_METRICS_PORT', 9090))

    def _parse_origins(self, origins_str: str) -> List[str]:
        """CORS origins 파싱"""
        try:
            import json
            return json.loads(origins_str)
        except:
            return ["*"]

# Global variables
config = ProductionConfig()
react_framework: Optional[ReActFramework] = None
tool_manager: Optional[PACAToolManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global react_framework, tool_manager

    # Startup
    logger.info("Starting PACA Production Server", env=config.env)

    try:
        # Initialize tool manager
        tool_manager = PACAToolManager()

        # Register tools based on configuration
        if config.enable_web_search:
            web_search_tool = WebSearchTool()
            await tool_manager.register_tool(web_search_tool)
            logger.info("WebSearchTool registered")

        if config.enable_file_manager:
            file_manager_tool = FileManagerTool(sandbox_mode=config.sandbox_mode)
            await tool_manager.register_tool(file_manager_tool)
            logger.info("FileManagerTool registered")

        # Initialize ReAct framework
        react_framework = ReActFramework(tool_manager)

        logger.info("PACA system initialized successfully",
                   tools_count=len(tool_manager.tools))

        yield

        # Shutdown
        logger.info("Shutting down PACA Production Server")

        # Cleanup resources
        if tool_manager:
            # Close any open resources
            pass

    except Exception as e:
        logger.error("Failed to initialize PACA system", error=str(e))
        raise

# Create FastAPI app
app = FastAPI(
    title="PACA Production API",
    description="Production API for Personal Autonomous Cognitive Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """메트릭 수집 미들웨어"""
    if config.metrics_enabled:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()

        with REQUEST_DURATION.time():
            response = await call_next(request)

        return response
    else:
        return await call_next(request)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "PACA Production API",
        "version": "1.0.0",
        "status": "running",
        "environment": config.env,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # Check system components
        components = {
            "react_framework": react_framework is not None,
            "tool_manager": tool_manager is not None,
            "tools_registered": len(tool_manager.tools) if tool_manager else 0
        }

        all_healthy = all(components.values()) and components["tools_registered"] > 0

        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/v1/execute", response_model=PACAResponse)
async def execute_action(request: PACARequest, background_tasks: BackgroundTasks):
    """PACA 액션 실행"""
    try:
        if not react_framework:
            raise HTTPException(status_code=503, detail="PACA system not initialized")

        logger.info("Executing PACA action",
                   session_id=request.session_id,
                   action=request.action,
                   tool_name=request.tool_name)

        # Get or create session
        if request.session_id not in react_framework.sessions:
            session = await react_framework.create_session(request.session_id)
        else:
            session = react_framework.sessions[request.session_id]

        # Execute action based on type
        result = None
        if request.action == "think":
            result = await react_framework.think(
                session, request.content, request.confidence
            )
        elif request.action == "act":
            if not request.tool_name:
                raise HTTPException(status_code=400, detail="tool_name required for act action")

            result = await react_framework.act(
                session, request.tool_name, **request.tool_params
            )

            # Update tool execution metrics
            if config.metrics_enabled:
                status = "success" if result.tool_result and result.tool_result.success else "failure"
                TOOL_EXECUTION_COUNT.labels(tool_name=request.tool_name, status=status).inc()

        elif request.action == "observe":
            result = await react_framework.observe(session, request.content)
        elif request.action == "reflect":
            result = await react_framework.reflect(session, request.content)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

        # Prepare response
        response = PACAResponse(
            success=True,
            session_id=request.session_id,
            step_id=result.step_id,
            action=result.step_type.value,
            content=result.content,
            confidence=result.confidence,
            tool_result=result.tool_result.__dict__ if result.tool_result else None
        )

        logger.info("Action executed successfully",
                   session_id=request.session_id,
                   step_id=result.step_id,
                   action=result.step_type.value)

        return response

    except Exception as e:
        logger.error("Action execution failed",
                    session_id=request.session_id,
                    action=request.action,
                    error=str(e))

        return PACAResponse(
            success=False,
            session_id=request.session_id,
            step_id="",
            action=request.action,
            content="",
            error=str(e)
        )

@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """세션 정보 조회"""
    try:
        if not react_framework or session_id not in react_framework.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = react_framework.sessions[session_id]

        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "steps_count": len(session.steps),
            "status": session.status.value if hasattr(session, 'status') else "active"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tools")
async def list_tools():
    """등록된 도구 목록 조회"""
    try:
        if not tool_manager:
            raise HTTPException(status_code=503, detail="Tool manager not initialized")

        tools = []
        for tool in tool_manager.tools.values():
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "type": tool.tool_type.value,
                "description": getattr(tool, 'description', ''),
                "enabled": True
            })

        return {
            "tools": tools,
            "count": len(tools)
        }

    except Exception as e:
        logger.error("Failed to list tools", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus 메트릭 엔드포인트"""
    if not config.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """글로벌 예외 처리기"""
    logger.error("Unhandled exception",
                path=request.url.path,
                method=request.method,
                error=str(exc))

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.debug else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

def main():
    """메인 함수"""
    logger.info("Starting PACA Production Server",
               host=config.host,
               port=config.port,
               workers=config.workers)

    uvicorn.run(
        "production_server:app",
        host=config.host,
        port=config.port,
        workers=1,  # Use 1 worker for development, increase for production
        reload=False,
        log_level="info" if not config.debug else "debug"
    )

if __name__ == "__main__":
    main()
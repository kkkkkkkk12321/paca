"""
Image Generator Tool
Gemini API를 사용한 이미지 생성 도구
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging
import io
from PIL import Image
import base64
import os

# 조건부 임포트
try:
    from ..base import Tool, ToolResult, ToolError, ToolType
    from ...core.types.base import (
        ID, Timestamp, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.tools.base import Tool, ToolResult, ToolError, ToolType
    from paca.core.types.base import (
        ID, Timestamp, current_timestamp, generate_id, create_success, create_failure
    )

# Gemini API 임포트
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class ImageGenerationModel(Enum):
    """이미지 생성 모델"""
    GEMINI_NATIVE = "gemini-2.5-flash-image-preview"  # Gemini 네이티브 이미지 생성
    IMAGEN_3 = "imagen-3.0-generate-002"               # Imagen 3 고품질 이미지 생성


class AspectRatio(Enum):
    """가로세로 비율"""
    SQUARE = "1:1"
    PORTRAIT = "3:4"
    LANDSCAPE = "4:3"
    VERTICAL = "9:16"
    HORIZONTAL = "16:9"


class PersonGeneration(Enum):
    """인물 생성 설정"""
    ALLOW_ALL = "ALLOW_ALL"           # 모든 인물 허용 (유럽/MENA 제외)
    ALLOW_ADULT = "ALLOW_ADULT"       # 성인만 허용
    DONT_ALLOW = "DONT_ALLOW"         # 인물 생성 금지


@dataclass
class ImageGenerationRequest:
    """이미지 생성 요청"""
    request_id: ID
    prompt: str
    model: ImageGenerationModel
    number_of_images: int = 1
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    person_generation: PersonGeneration = PersonGeneration.ALLOW_ADULT
    negative_prompt: Optional[str] = None
    style_guidance: Optional[str] = None
    quality_level: Optional[str] = None
    created_at: Timestamp = field(default_factory=current_timestamp)


@dataclass
class GeneratedImage:
    """생성된 이미지"""
    image_id: ID
    image_data: bytes
    mime_type: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Timestamp = field(default_factory=current_timestamp)


@dataclass
class ImageGenerationResult:
    """이미지 생성 결과"""
    result_id: ID
    request: ImageGenerationRequest
    images: List[GeneratedImage]
    success: bool
    error_message: Optional[str] = None
    generation_time: float = 0.0
    total_cost: Optional[float] = None
    completed_at: Timestamp = field(default_factory=current_timestamp)


class ImageGenerator(Tool):
    """
    Gemini API를 사용한 이미지 생성 도구
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="image_generator",
            tool_type=ToolType.GENERATION,
            description="Gemini API를 사용하여 텍스트 프롬프트로부터 이미지를 생성합니다"
        )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.generation_history: List[ImageGenerationResult] = []
        self.output_directory = "generated_images"

        # 디렉토리 생성
        os.makedirs(self.output_directory, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        if not GEMINI_AVAILABLE:
            self.logger.error("Gemini API 라이브러리가 설치되지 않았습니다. 'pip install google-genai' 명령어로 설치하세요.")
            raise ImportError("google-genai 패키지가 필요합니다")

        if not self.api_key:
            self.logger.error("Gemini API 키가 제공되지 않았습니다. GEMINI_API_KEY 환경변수를 설정하세요.")
            raise ValueError("Gemini API 키가 필요합니다")

        # 클라이언트 초기화
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.logger.info("Gemini API 클라이언트 초기화 완료")
        except Exception as e:
            self.logger.error(f"Gemini API 클라이언트 초기화 실패: {e}")
            raise

    def validate_input(self, **params) -> bool:
        """입력 매개변수 검증"""
        try:
            prompt = params.get('prompt')
            if not prompt or not isinstance(prompt, str):
                return False

            if len(prompt.strip()) == 0:
                return False

            # 프롬프트 길이 제한
            if len(prompt) > 2000:
                return False

            # 이미지 수 검증
            number_of_images = params.get('number_of_images', 1)
            if not isinstance(number_of_images, int) or number_of_images < 1 or number_of_images > 4:
                return False

            # 모델 검증
            model = params.get('model', ImageGenerationModel.GEMINI_NATIVE.value)
            if isinstance(model, str):
                try:
                    ImageGenerationModel(model)
                except ValueError:
                    return False
            elif not isinstance(model, ImageGenerationModel):
                return False

            return True

        except Exception as e:
            self.logger.error(f"입력 검증 실패: {e}")
            return False

    async def execute(self, **kwargs) -> ToolResult:
        """이미지 생성 실행"""
        start_time = current_timestamp()

        try:
            if not self.validate_input(**kwargs):
                return ToolResult(
                    success=False,
                    error="유효하지 않은 입력 매개변수입니다"
                )

            # 요청 객체 생성
            request = self._create_request(**kwargs)

            # 이미지 생성
            if request.model == ImageGenerationModel.GEMINI_NATIVE:
                result = await self._generate_with_gemini_native(request)
            elif request.model == ImageGenerationModel.IMAGEN_3:
                result = await self._generate_with_imagen3(request)
            else:
                return ToolResult(
                    success=False,
                    error=f"지원하지 않는 모델: {request.model.value}"
                )

            # 생성 시간 계산
            result.generation_time = current_timestamp() - start_time

            # 결과 저장
            self.generation_history.append(result)
            if len(self.generation_history) > 100:
                self.generation_history = self.generation_history[-100:]

            if result.success:
                response_data = {
                    'result_id': result.result_id,
                    'images_generated': len(result.images),
                    'generation_time': result.generation_time,
                    'images': [
                        {
                            'image_id': img.image_id,
                            'file_path': img.file_path,
                            'mime_type': img.mime_type,
                            'metadata': img.metadata
                        }
                        for img in result.images
                    ]
                }

                return ToolResult(
                    success=True,
                    data=response_data,
                    metadata={
                        'prompt': request.prompt,
                        'model': request.model.value,
                        'aspect_ratio': request.aspect_ratio.value,
                        'generation_time': result.generation_time
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=result.error_message or "이미지 생성 실패"
                )

        except Exception as e:
            self.logger.error(f"이미지 생성 실행 실패: {e}")
            return ToolResult(
                success=False,
                error=f"이미지 생성 중 오류 발생: {str(e)}"
            )

    def _create_request(self, **kwargs) -> ImageGenerationRequest:
        """요청 객체 생성"""
        prompt = kwargs['prompt']
        model_value = kwargs.get('model', ImageGenerationModel.GEMINI_NATIVE.value)

        if isinstance(model_value, str):
            model = ImageGenerationModel(model_value)
        else:
            model = model_value

        aspect_ratio_value = kwargs.get('aspect_ratio', AspectRatio.SQUARE.value)
        if isinstance(aspect_ratio_value, str):
            aspect_ratio = AspectRatio(aspect_ratio_value)
        else:
            aspect_ratio = aspect_ratio_value

        person_generation_value = kwargs.get('person_generation', PersonGeneration.ALLOW_ADULT.value)
        if isinstance(person_generation_value, str):
            person_generation = PersonGeneration(person_generation_value)
        else:
            person_generation = person_generation_value

        return ImageGenerationRequest(
            request_id=generate_id("img_req_"),
            prompt=prompt,
            model=model,
            number_of_images=kwargs.get('number_of_images', 1),
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            negative_prompt=kwargs.get('negative_prompt'),
            style_guidance=kwargs.get('style_guidance'),
            quality_level=kwargs.get('quality_level')
        )

    async def _generate_with_gemini_native(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Gemini 네이티브 이미지 생성"""
        try:
            self.logger.info(f"Gemini 네이티브로 이미지 생성 시작: {request.prompt}")

            # 향상된 프롬프트 생성
            enhanced_prompt = self._enhance_prompt(request)

            # Gemini 네이티브 이미지 생성 API 호출
            response = self.client.models.generate_content(
                model=request.model.value,
                contents=enhanced_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            generated_images = []

            # 응답에서 이미지 추출
            for i, part in enumerate(response.candidates[0].content.parts):
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type

                    # 이미지 저장
                    image_id = generate_id("img_")
                    file_extension = mime_type.split('/')[-1]
                    file_path = os.path.join(self.output_directory, f"{image_id}.{file_extension}")

                    with open(file_path, 'wb') as f:
                        f.write(image_data)

                    generated_image = GeneratedImage(
                        image_id=image_id,
                        image_data=image_data,
                        mime_type=mime_type,
                        file_path=file_path,
                        metadata={
                            'model': request.model.value,
                            'prompt': request.prompt,
                            'aspect_ratio': request.aspect_ratio.value
                        }
                    )

                    generated_images.append(generated_image)

            if not generated_images:
                return ImageGenerationResult(
                    result_id=generate_id("result_"),
                    request=request,
                    images=[],
                    success=False,
                    error_message="생성된 이미지가 없습니다"
                )

            self.logger.info(f"Gemini 네이티브 이미지 생성 완료: {len(generated_images)}개 이미지")

            return ImageGenerationResult(
                result_id=generate_id("result_"),
                request=request,
                images=generated_images,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Gemini 네이티브 이미지 생성 실패: {e}")
            return ImageGenerationResult(
                result_id=generate_id("result_"),
                request=request,
                images=[],
                success=False,
                error_message=str(e)
            )

    async def _generate_with_imagen3(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Imagen 3를 사용한 고품질 이미지 생성"""
        try:
            self.logger.info(f"Imagen 3로 이미지 생성 시작: {request.prompt}")

            # 향상된 프롬프트 생성
            enhanced_prompt = self._enhance_prompt(request)

            # Imagen 3 API 호출
            response = self.client.models.generate_images(
                model=request.model.value,
                prompt=enhanced_prompt,
                config={
                    'number_of_images': request.number_of_images,
                    'aspect_ratio': request.aspect_ratio.value,
                    'person_generation': request.person_generation.value,
                    'output_mime_type': 'image/jpeg'
                }
            )

            generated_images = []

            # 응답에서 이미지 추출
            for i, generated_image in enumerate(response.generated_images):
                image_data = generated_image.image.image_bytes
                mime_type = "image/jpeg"

                # 이미지 저장
                image_id = generate_id("img_")
                file_path = os.path.join(self.output_directory, f"{image_id}.jpg")

                # PIL을 사용해 이미지 저장
                image = Image.open(io.BytesIO(image_data))
                image.save(file_path, 'JPEG', quality=95)

                generated_img = GeneratedImage(
                    image_id=image_id,
                    image_data=image_data,
                    mime_type=mime_type,
                    file_path=file_path,
                    metadata={
                        'model': request.model.value,
                        'prompt': request.prompt,
                        'aspect_ratio': request.aspect_ratio.value,
                        'person_generation': request.person_generation.value
                    }
                )

                generated_images.append(generated_img)

            self.logger.info(f"Imagen 3 이미지 생성 완료: {len(generated_images)}개 이미지")

            return ImageGenerationResult(
                result_id=generate_id("result_"),
                request=request,
                images=generated_images,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Imagen 3 이미지 생성 실패: {e}")
            return ImageGenerationResult(
                result_id=generate_id("result_"),
                request=request,
                images=[],
                success=False,
                error_message=str(e)
            )

    def _enhance_prompt(self, request: ImageGenerationRequest) -> str:
        """프롬프트 향상"""
        prompt = request.prompt.strip()

        # 스타일 가이던스 추가
        if request.style_guidance:
            prompt += f", {request.style_guidance}"

        # 품질 수정자 추가
        quality_modifiers = []

        if request.quality_level == "high":
            quality_modifiers.extend([
                "high quality", "detailed", "professional"
            ])
        elif request.quality_level == "artistic":
            quality_modifiers.extend([
                "artistic", "creative", "beautiful composition"
            ])
        elif request.quality_level == "photorealistic":
            quality_modifiers.extend([
                "photorealistic", "highly detailed", "professional photography"
            ])

        if quality_modifiers:
            prompt += f", {', '.join(quality_modifiers)}"

        # 네거티브 프롬프트 처리 (Imagen 3에서 지원)
        if request.negative_prompt and request.model == ImageGenerationModel.IMAGEN_3:
            prompt += f" [NEGATIVE: {request.negative_prompt}]"

        return prompt

    def get_generation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """생성 히스토리 조회"""
        recent_results = self.generation_history[-limit:] if limit else self.generation_history

        return [
            {
                'result_id': result.result_id,
                'prompt': result.request.prompt,
                'model': result.request.model.value,
                'images_count': len(result.images),
                'success': result.success,
                'generation_time': result.generation_time,
                'completed_at': result.completed_at,
                'error_message': result.error_message
            }
            for result in recent_results
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """생성 통계"""
        if not self.generation_history:
            return {'total_generations': 0}

        total_generations = len(self.generation_history)
        successful_generations = sum(1 for r in self.generation_history if r.success)
        total_images = sum(len(r.images) for r in self.generation_history)

        avg_generation_time = sum(r.generation_time for r in self.generation_history) / total_generations

        model_usage = {}
        for result in self.generation_history:
            model = result.request.model.value
            model_usage[model] = model_usage.get(model, 0) + 1

        return {
            'total_generations': total_generations,
            'successful_generations': successful_generations,
            'success_rate': successful_generations / total_generations,
            'total_images_generated': total_images,
            'average_generation_time': avg_generation_time,
            'model_usage': model_usage,
            'output_directory': self.output_directory
        }

    def clear_history(self):
        """히스토리 클리어"""
        self.generation_history.clear()
        self.logger.info("이미지 생성 히스토리가 클리어되었습니다")

    def set_output_directory(self, directory: str):
        """출력 디렉토리 설정"""
        self.output_directory = directory
        os.makedirs(directory, exist_ok=True)
        self.logger.info(f"출력 디렉토리 설정: {directory}")


# 전역 인스턴스 (싱글톤 패턴)
_image_generator_instance = None


def get_image_generator(api_key: Optional[str] = None) -> ImageGenerator:
    """이미지 생성기 싱글톤 인스턴스 획득"""
    global _image_generator_instance
    if _image_generator_instance is None:
        _image_generator_instance = ImageGenerator(api_key)
    return _image_generator_instance


# 편의 함수
async def generate_image(prompt: str,
                        model: Union[str, ImageGenerationModel] = ImageGenerationModel.GEMINI_NATIVE,
                        number_of_images: int = 1,
                        aspect_ratio: Union[str, AspectRatio] = AspectRatio.SQUARE,
                        person_generation: Union[str, PersonGeneration] = PersonGeneration.ALLOW_ADULT,
                        quality_level: Optional[str] = None,
                        style_guidance: Optional[str] = None,
                        negative_prompt: Optional[str] = None,
                        api_key: Optional[str] = None) -> ToolResult:
    """
    간편한 이미지 생성 함수

    Args:
        prompt: 이미지 생성 프롬프트
        model: 사용할 모델 (gemini-2.5-flash-image-preview 또는 imagen-3.0-generate-002)
        number_of_images: 생성할 이미지 수 (1-4)
        aspect_ratio: 가로세로 비율
        person_generation: 인물 생성 설정
        quality_level: 품질 수준 ("high", "artistic", "photorealistic")
        style_guidance: 스타일 가이던스
        negative_prompt: 네거티브 프롬프트 (Imagen 3만 지원)
        api_key: Gemini API 키

    Returns:
        ToolResult: 이미지 생성 결과
    """
    generator = get_image_generator(api_key)

    return await generator.execute(
        prompt=prompt,
        model=model,
        number_of_images=number_of_images,
        aspect_ratio=aspect_ratio,
        person_generation=person_generation,
        quality_level=quality_level,
        style_guidance=style_guidance,
        negative_prompt=negative_prompt
    )
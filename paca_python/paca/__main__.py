"""
PACA v5 Python Edition - Main Entry Point
Personal Adaptive Cognitive Assistant v5
"""

import asyncio
import json
import sys
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional
import paca.cognitive._enable_collab_patch

try:  # Optional dependency (only needed for YAML overrides)
    import yaml  # type: ignore
except Exception:  # pragma: no cover - YAML support is optional at runtime
    yaml = None

# UTF-8 인코딩 설정 (Windows 호환성)
if os.name == 'nt':  # Windows
    import locale
    try:
        # Python 3.7+ 에서 UTF-8 모드 활성화
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # 이전 버전 Python 지원
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    # 콘솔 코드페이지를 UTF-8로 설정
    try:
        os.system('chcp 65001')
    except Exception:
        pass

from .system import PacaSystem, PacaConfig
from .core.utils.logger import StructuredLogger


def create_parser() -> argparse.ArgumentParser:
    """CLI 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        prog="paca",
        description="PACA v5 - Personal Adaptive Cognitive Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paca --interactive           대화형 모드로 실행
  paca --gui                   GUI 애플리케이션 실행
  paca --message "안녕하세요"   단일 메시지 처리
  paca --config config.json    설정 파일 지정
  paca --version               버전 정보 표시
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="PACA v5.0.0"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드로 실행"
    )

    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="GUI 애플리케이션 실행"
    )

    parser.add_argument(
        "--message", "-m",
        type=str,
        help="처리할 단일 메시지"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="설정 파일 경로"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="디버그 모드 활성화"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨 설정"
    )

    return parser


async def run_interactive_mode(paca_system: PacaSystem):
    """대화형 모드 실행"""
    print("🤖 PACA v5 대화형 모드")
    print("종료하려면 'quit', 'exit', 또는 Ctrl+C'를 입력하세요.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 안녕히 가세요!")
                    break

                if not user_input:
                    continue

                # 메시지 처리
                result = await paca_system.process_message(user_input)

                if result.is_success:
                    response = result.data.get("response", "응답을 생성할 수 없습니다.")
                    print(f"PACA: {response}")

                    # 성능 정보 표시 (디버그 모드에서만)
                    if hasattr(paca_system.config, 'debug') and paca_system.config.debug:
                        processing_time = result.data.get("processing_time", 0)
                        confidence = result.data.get("confidence", 0)
                        print(f"       (처리시간: {processing_time:.3f}s, 신뢰도: {confidence:.2f})")

                else:
                    print(f"❌ 오류: {result.error}")

                print()

            except KeyboardInterrupt:
                print("\n👋 안녕히 가세요!")
                break
            except EOFError:
                print("\n👋 안녕히 가세요!")
                break

    except Exception as e:
        print(f"❌ 대화형 모드 실행 중 오류: {str(e)}")


async def process_single_message(paca_system: PacaSystem, message: str):
    """단일 메시지 처리"""
    try:
        result = await paca_system.process_message(message)

        if result.is_success:
            response = result.data.get("response", "응답을 생성할 수 없습니다.")
            print(response)

            # 상세 정보 (옵션)
            processing_time = result.data.get("processing_time", 0)
            confidence = result.data.get("confidence", 0)
            if processing_time > 0:
                print(f"\n처리시간: {processing_time:.3f}s, 신뢰도: {confidence:.2f}", file=sys.stderr)

        else:
            print(f"오류: {result.error}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"메시지 처리 중 오류: {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_gui():
    """GUI 애플리케이션 실행"""
    try:
        from desktop_app.main import main as gui_main
        print("🖥️ GUI 애플리케이션을 시작합니다...")
        gui_main()
    except ImportError:
        print("❌ GUI 의존성이 설치되지 않았습니다.")
        print("다음 명령으로 설치하세요: pip install 'paca[gui]'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ GUI 실행 중 오류: {str(e)}")
        sys.exit(1)


async def main_async():
    """비동기 메인 함수"""
    parser = create_parser()
    args = parser.parse_args()

    # 로거 설정
    logger = StructuredLogger("PacaMain")

    try:
        # 설정 생성
        config = PacaConfig()

        if args.config:
            try:
                overrides = _load_user_config(args.config)
            except FileNotFoundError:
                parser.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
            except ValueError as config_error:
                parser.error(str(config_error))
            except Exception as unexpected_error:  # pragma: no cover - safety net
                parser.error(f"설정 파일을 읽는 중 오류가 발생했습니다: {unexpected_error}")
            else:
                _apply_overrides(config, overrides)

        # === 추가: 협업 재시도 정책 JSON 로드 ===
        from paca.cognitive._collab_policy_loader import load_policy, apply_to_config
        policy = load_policy()
        apply_to_config(config, policy)
        # =======================================

        # === 추가: 현재 임계값들 디버그 프린트 ===
        try:
            print("[CFG] thresholds:",
                  "reasoning=", getattr(config, "reasoning_confidence_threshold", None),
                  "backtrack=", getattr(config, "backtrack_confidence_threshold", None),
                  "switch=", getattr(config, "strategy_switch_confidence_threshold", None),
                  "escal_min=", (getattr(config, "escalation", {}) or {}).get("min_confidence"))
        except Exception:
            pass
        # =======================================

        if args.debug:
            config.log_level = "DEBUG"
            setattr(config, "debug", True)
        else:
            config.log_level = args.log_level
            setattr(config, "debug", False)

        # GUI 모드
        if args.gui:
            run_gui()
            return

        # PACA 시스템 초기화
        print("🚀 PACA v5 시스템을 초기화하는 중...")
        paca_system = PacaSystem(config)

        result = await paca_system.initialize()
        if not result.is_success:
            print(f"❌ 시스템 초기화 실패: {result.error}")
            sys.exit(1)

        print("✅ PACA v5 시스템 준비 완료!\n")

        # 모드별 실행
        if args.message:
            # 단일 메시지 모드
            await process_single_message(paca_system, args.message)

        elif args.interactive:
            # 대화형 모드
            await run_interactive_mode(paca_system)

        else:
            # 기본: 도움말 표시
            parser.print_help()

        # 시스템 정리
        await paca_system.cleanup()

    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"메인 실행 중 오류: {str(e)}")
        print(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
        sys.exit(1)


def main():
    """메인 진입점"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ 실행 중 치명적 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


def _load_user_config(path: Path) -> Dict[str, Any]:
    """Load a user-specified configuration file."""

    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.is_dir():
        raise ValueError("설정 파일 경로가 디렉터리입니다. 파일을 지정해 주세요.")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ValueError("YAML 설정을 로드하려면 PyYAML이 필요합니다. 'pip install pyyaml'을 실행하세요.")

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise ValueError("지원하지 않는 설정 파일 형식입니다. JSON 또는 YAML 파일을 사용하세요.")

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError("설정 파일의 최상위 구조는 객체(JSON) 또는 매핑(YAML)이어야 합니다.")

    return data


def _apply_overrides(config: PacaConfig, overrides: Dict[str, Any]) -> None:
    """Merge user overrides into the runtime configuration."""

    def _set_attr(name: str, value: Any) -> bool:
        if hasattr(config, name):
            setattr(config, name, value)
            return True
        return False

    for key, value in overrides.items():
        if key == "llm" and isinstance(value, dict):
            api_keys = value.get("api_keys")
            if isinstance(api_keys, (list, tuple)):
                cleaned = [str(item).strip() for item in api_keys if str(item).strip()]
                if cleaned:
                    config.gemini_api_keys = cleaned

            default_model = value.get("default_model")
            if isinstance(default_model, str):
                config.default_llm_model = default_model

            temperature = value.get("temperature")
            if temperature is not None:
                try:
                    config.llm_temperature = float(temperature)
                except (TypeError, ValueError):
                    pass

            max_tokens = value.get("max_tokens")
            if max_tokens is not None:
                try:
                    config.llm_max_tokens = int(max_tokens)
                except (TypeError, ValueError):
                    pass

            timeout = value.get("timeout")
            if timeout is not None:
                try:
                    config.llm_timeout = float(timeout)
                except (TypeError, ValueError):
                    pass

            enable_cache = value.get("enable_caching")
            if enable_cache is not None:
                config.enable_llm_caching = bool(enable_cache)

            rotation = value.get("rotation")
            if isinstance(rotation, dict):
                strategy = rotation.get("strategy")
                if isinstance(strategy, str) and strategy.strip():
                    config.llm_rotation_strategy = strategy.strip()

                min_interval = rotation.get("min_interval_seconds")
                if min_interval is not None:
                    try:
                        config.llm_rotation_min_interval = float(min_interval)
                    except (TypeError, ValueError):
                        pass

            models = value.get("models")
            if isinstance(models, dict):
                config.llm_model_preferences = {
                    str(model_key): [str(item) for item in items]
                    for model_key, items in models.items()
                    if isinstance(items, (list, tuple))
                }

            continue

        if isinstance(value, dict):
            existing = getattr(config, key, None)
            if isinstance(existing, dict):
                existing.update(value)
            else:
                setattr(config, key, value)
            continue

        if not _set_attr(key, value):
            setattr(config, key, value)

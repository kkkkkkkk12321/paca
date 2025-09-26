# PACA v5 의존성 분석 리포트
============================================================

## 📊 전체 통계
- **총 모듈 수**: 54
- **내부 의존성**: 135
- **외부 의존성**: 657
- **순환 의존성**: 0

## 🏗️ 모듈별 의존성
| 모듈 | Fan-in | Fan-out | 내부 의존성 | 외부 의존성 |
|------|--------|---------|-------------|-------------|
| __main__ | 0 | 2 | 2 | 11 |
| api.base | 0 | 3 | 3 | 18 |
| cognitive.base | 1 | 4 | 4 | 14 |
| cognitive.learning | 0 | 4 | 4 | 10 |
| cognitive.memory.episodic | 0 | 3 | 3 | 12 |
| cognitive.memory.longterm | 0 | 3 | 3 | 15 |
| cognitive.memory.types | 0 | 1 | 1 | 3 |
| cognitive.memory.working | 0 | 3 | 3 | 11 |
| cognitive.models.actr | 0 | 3 | 3 | 16 |
| cognitive.models.base | 0 | 3 | 3 | 9 |
| cognitive.models.soar | 0 | 3 | 3 | 11 |
| config.base | 0 | 2 | 2 | 16 |
| controllers.execution | 0 | 3 | 3 | 15 |
| controllers.main | 0 | 7 | 7 | 16 |
| controllers.sentiment | 0 | 3 | 3 | 22 |
| controllers.validation | 0 | 3 | 3 | 30 |
| core.constants | 1 | 1 | 1 | 1 |
| core.constants.config | 0 | 0 | 0 | 3 |
| core.constants.limits | 0 | 0 | 0 | 1 |
| core.constants.messages | 0 | 0 | 0 | 5 |
| core.constants.paths | 0 | 0 | 0 | 4 |
| core.errors | 14 | 1 | 1 | 10 |
| core.errors.base | 10 | 1 | 1 | 15 |
| core.errors.cognitive | 1 | 1 | 1 | 4 |
| core.errors.reasoning | 1 | 1 | 1 | 3 |
| core.errors.validation | 0 | 2 | 2 | 2 |
| core.events | 8 | 2 | 2 | 14 |
| core.events.base | 2 | 1 | 1 | 13 |
| core.events.emitter | 4 | 2 | 2 | 17 |
| core.events.handlers | 0 | 2 | 2 | 13 |
| core.events.queue | 0 | 2 | 2 | 14 |
| core.types | 20 | 0 | 0 | 6 |
| core.types.base | 12 | 0 | 0 | 6 |
| core.utils | 2 | 2 | 2 | 29 |
| core.utils.async_utils | 2 | 0 | 0 | 14 |
| core.utils.logger | 12 | 1 | 1 | 18 |
| core.utils.math_utils | 0 | 0 | 0 | 7 |
| core.validators | 0 | 2 | 2 | 14 |
| data.base | 0 | 2 | 2 | 7 |
| learning.auto.engine | 0 | 4 | 4 | 28 |
| learning.auto.types | 0 | 1 | 1 | 6 |
| learning.memory.storage | 0 | 3 | 3 | 12 |
| learning.patterns.analyzer | 0 | 3 | 3 | 15 |
| learning.patterns.detector | 0 | 2 | 2 | 14 |
| mathematics.calculator | 0 | 2 | 2 | 8 |
| reasoning.base | 0 | 4 | 4 | 11 |
| services.analytics | 0 | 5 | 5 | 14 |
| services.auth | 0 | 5 | 5 | 23 |
| services.base | 0 | 4 | 4 | 12 |
| services.knowledge | 0 | 5 | 5 | 19 |
| services.learning | 0 | 5 | 5 | 11 |
| services.memory | 0 | 4 | 4 | 16 |
| services.notification | 0 | 5 | 5 | 11 |
| system | 1 | 10 | 10 | 8 |

## 💪 의존성 강도 분석
**높은 결합도 모듈:**
- core.types (결합도: 20)
- core.errors (결합도: 15)
- core.utils.logger (결합도: 13)
- core.types.base (결합도: 12)
- system (결합도: 11)

## 💡 개선 권장사항
- **결합도 감소**: 높은 결합도 모듈의 책임 분리 고려
- **고립된 모듈 검토**: core.constants.config, core.constants.limits, core.constants.messages, core.constants.paths, core.utils.math_utils
- **인터페이스 설계**: 모듈 간 명확한 인터페이스 정의
- **의존성 주입**: 런타임 의존성 구성으로 유연성 향상
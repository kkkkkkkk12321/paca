"""
피드백 분석기
수집된 피드백 데이터를 분석하고 인사이트를 제공
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import statistics

from .models import (
    FeedbackModel, FeedbackType, FeedbackStatus, SentimentScore,
    FeedbackStats, FeedbackAnalysis, UserSession
)
from .storage import FeedbackStorage

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """피드백 분석기 클래스"""

    def __init__(self, storage: FeedbackStorage):
        """
        초기화

        Args:
            storage: 피드백 저장소
        """
        self.storage = storage

    async def analyze_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_predictions: bool = True
    ) -> FeedbackAnalysis:
        """종합 피드백 분석"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()

            logger.info(f"Analyzing feedback from {start_date} to {end_date}")

            # 기본 통계 수집
            stats = await self.storage.get_feedback_stats(start_date, end_date)

            # 피드백 데이터 수집
            feedback_list = await self.storage.list_feedback(
                start_date=start_date,
                end_date=end_date,
                limit=10000  # 대량 데이터 분석
            )

            # 상세 분석 수행
            key_insights = await self._generate_insights(feedback_list, stats)
            improvement_suggestions = await self._generate_improvements(feedback_list, stats)
            critical_issues = await self._identify_critical_issues(feedback_list)

            # 도구별 성능 분석
            tool_performance = await self._analyze_tool_performance(feedback_list)

            # 일반적인 불만사항 및 긍정적 피드백
            common_complaints = await self._analyze_complaints(feedback_list)
            positive_highlights = await self._analyze_positive_feedback(feedback_list)

            # 예측 (선택적)
            trend_predictions = {}
            recommended_actions = []
            if include_predictions:
                trend_predictions = await self._predict_trends(feedback_list, stats)
                recommended_actions = await self._recommend_actions(
                    feedback_list, stats, critical_issues
                )

            analysis = FeedbackAnalysis(
                period_start=start_date,
                period_end=end_date,
                stats=stats,
                key_insights=key_insights,
                improvement_suggestions=improvement_suggestions,
                critical_issues=critical_issues,
                tool_performance=tool_performance,
                common_complaints=common_complaints,
                positive_highlights=positive_highlights,
                trend_predictions=trend_predictions,
                recommended_actions=recommended_actions
            )

            logger.info(f"Feedback analysis completed: {analysis.analysis_id}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            raise

    async def _generate_insights(
        self,
        feedback_list: List[FeedbackModel],
        stats: FeedbackStats
    ) -> List[str]:
        """핵심 인사이트 생성"""
        insights = []

        try:
            # 피드백 볼륨 인사이트
            if stats.total_feedback > 0:
                insights.append(f"총 {stats.total_feedback}개의 피드백이 수집되었습니다.")

                # 평점 인사이트
                if stats.average_rating:
                    if stats.average_rating >= 4.0:
                        insights.append(f"평균 평점 {stats.average_rating:.1f}점으로 사용자 만족도가 높습니다.")
                    elif stats.average_rating <= 2.0:
                        insights.append(f"평균 평점 {stats.average_rating:.1f}점으로 사용자 만족도 개선이 필요합니다.")
                    else:
                        insights.append(f"평균 평점 {stats.average_rating:.1f}점으로 보통 수준입니다.")

                # 감정 분포 인사이트
                total_sentiment = sum(stats.sentiment_distribution.values())
                if total_sentiment > 0:
                    positive_ratio = stats.sentiment_distribution.get('POSITIVE', 0) / total_sentiment
                    negative_ratio = stats.sentiment_distribution.get('NEGATIVE', 0) / total_sentiment

                    if positive_ratio > 0.6:
                        insights.append("사용자 감정이 전반적으로 긍정적입니다.")
                    elif negative_ratio > 0.4:
                        insights.append("부정적 감정의 피드백이 상당히 많습니다.")

                # 피드백 타입별 인사이트
                most_common_type = max(stats.feedback_by_type.items(), key=lambda x: x[1])
                insights.append(f"가장 많은 피드백 유형은 '{most_common_type[0]}'입니다 ({most_common_type[1]}건).")

                # 해결률 인사이트
                resolved_count = stats.feedback_by_status.get('resolved', 0)
                if stats.total_feedback > 0:
                    resolution_rate = resolved_count / stats.total_feedback
                    if resolution_rate < 0.5:
                        insights.append(f"피드백 해결률이 {resolution_rate:.1%}로 낮습니다.")

            else:
                insights.append("분석 기간 동안 수집된 피드백이 없습니다.")

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            insights.append("인사이트 생성 중 오류가 발생했습니다.")

        return insights

    async def _generate_improvements(
        self,
        feedback_list: List[FeedbackModel],
        stats: FeedbackStats
    ) -> List[str]:
        """개선 제안 생성"""
        improvements = []

        try:
            # 낮은 평점 개선
            if stats.average_rating and stats.average_rating < 3.0:
                improvements.append("전체적인 사용자 경험 개선이 필요합니다.")

            # 도구 실행 문제 개선
            tool_issues = [f for f in feedback_list
                          if f.feedback_type == FeedbackType.TOOL_EXECUTION and f.rating and f.rating <= 2]
            if len(tool_issues) > 5:
                improvements.append("도구 실행 안정성 개선이 필요합니다.")

            # 성능 문제 개선
            perf_issues = [f for f in feedback_list
                          if f.feedback_type == FeedbackType.PERFORMANCE and f.rating and f.rating <= 2]
            if len(perf_issues) > 3:
                improvements.append("시스템 성능 최적화가 필요합니다.")

            # 미해결 피드백 처리
            unresolved_count = sum(count for status, count in stats.feedback_by_status.items()
                                 if status in ['pending', 'reviewed'])
            if unresolved_count > stats.total_feedback * 0.3:
                improvements.append("미해결 피드백 처리 프로세스 개선이 필요합니다.")

            # 부정적 감정 개선
            negative_count = stats.sentiment_distribution.get('NEGATIVE', 0)
            total_sentiment = sum(stats.sentiment_distribution.values())
            if total_sentiment > 0 and negative_count / total_sentiment > 0.3:
                improvements.append("사용자 감정 개선을 위한 UX 향상이 필요합니다.")

        except Exception as e:
            logger.error(f"Failed to generate improvements: {e}")

        return improvements

    async def _identify_critical_issues(
        self,
        feedback_list: List[FeedbackModel]
    ) -> List[str]:
        """중요한 이슈 식별"""
        critical_issues = []

        try:
            # 최근 24시간 내 심각한 문제
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_critical = [
                f for f in feedback_list
                if (f.timestamp > recent_cutoff and
                    f.feedback_type in [FeedbackType.BUG_REPORT, FeedbackType.TOOL_EXECUTION] and
                    f.rating and f.rating <= 2)
            ]

            if len(recent_critical) > 3:
                critical_issues.append(f"최근 24시간 동안 {len(recent_critical)}건의 심각한 문제가 보고되었습니다.")

            # 반복되는 오류 패턴
            error_patterns = defaultdict(int)
            for feedback in feedback_list:
                if (feedback.context and feedback.context.error_message and
                    feedback.feedback_type == FeedbackType.TOOL_EXECUTION):
                    # 오류 메시지의 첫 번째 줄만 사용
                    error_key = feedback.context.error_message.split('\n')[0][:100]
                    error_patterns[error_key] += 1

            for error, count in error_patterns.items():
                if count >= 5:  # 5회 이상 반복되는 오류
                    critical_issues.append(f"반복 오류 '{error[:50]}...' ({count}회)")

            # 도구별 실패율 체크
            tool_failures = defaultdict(lambda: {'total': 0, 'failed': 0})
            for feedback in feedback_list:
                if feedback.context and feedback.context.tool_name:
                    tool_name = feedback.context.tool_name
                    tool_failures[tool_name]['total'] += 1
                    if feedback.context.success is False:
                        tool_failures[tool_name]['failed'] += 1

            for tool, data in tool_failures.items():
                if data['total'] >= 10:  # 충분한 샘플
                    failure_rate = data['failed'] / data['total']
                    if failure_rate > 0.3:  # 30% 이상 실패
                        critical_issues.append(f"도구 '{tool}' 실패율 {failure_rate:.1%}")

        except Exception as e:
            logger.error(f"Failed to identify critical issues: {e}")

        return critical_issues

    async def _analyze_tool_performance(
        self,
        feedback_list: List[FeedbackModel]
    ) -> Dict[str, Dict[str, Any]]:
        """도구별 성능 분석"""
        tool_performance = defaultdict(lambda: {
            'total_usage': 0,
            'success_count': 0,
            'failure_count': 0,
            'average_rating': 0.0,
            'average_execution_time': 0.0,
            'ratings': [],
            'execution_times': []
        })

        try:
            for feedback in feedback_list:
                if not feedback.context or not feedback.context.tool_name:
                    continue

                tool_name = feedback.context.tool_name
                perf = tool_performance[tool_name]

                perf['total_usage'] += 1

                if feedback.context.success is True:
                    perf['success_count'] += 1
                elif feedback.context.success is False:
                    perf['failure_count'] += 1

                if feedback.rating:
                    perf['ratings'].append(feedback.rating)

                if feedback.context.execution_time:
                    perf['execution_times'].append(feedback.context.execution_time)

            # 통계 계산
            for tool_name, perf in tool_performance.items():
                if perf['ratings']:
                    perf['average_rating'] = statistics.mean(perf['ratings'])

                if perf['execution_times']:
                    perf['average_execution_time'] = statistics.mean(perf['execution_times'])

                perf['success_rate'] = (
                    perf['success_count'] / perf['total_usage']
                    if perf['total_usage'] > 0 else 0
                )

                # 리스트는 JSON 직렬화를 위해 제거
                del perf['ratings']
                del perf['execution_times']

        except Exception as e:
            logger.error(f"Failed to analyze tool performance: {e}")

        return dict(tool_performance)

    async def _analyze_complaints(
        self,
        feedback_list: List[FeedbackModel]
    ) -> List[Dict[str, Any]]:
        """일반적인 불만사항 분석"""
        complaints = []

        try:
            # 낮은 평점의 피드백 수집
            negative_feedback = [
                f for f in feedback_list
                if (f.rating and f.rating <= 2) or
                   (f.sentiment_score and f.sentiment_score in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE])
            ]

            # 불만 유형별 분류
            complaint_types = defaultdict(list)
            for feedback in negative_feedback:
                complaint_types[feedback.feedback_type].append(feedback)

            # 상위 불만사항 생성
            for feedback_type, feedback_items in complaint_types.items():
                if len(feedback_items) >= 3:  # 3건 이상인 경우만
                    complaint = {
                        'type': feedback_type.value,
                        'count': len(feedback_items),
                        'average_rating': statistics.mean([f.rating for f in feedback_items if f.rating]),
                        'sample_feedback': [
                            f.text_feedback for f in feedback_items[:3]
                            if f.text_feedback
                        ]
                    }
                    complaints.append(complaint)

            # 빈도순 정렬
            complaints.sort(key=lambda x: x['count'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to analyze complaints: {e}")

        return complaints[:5]  # 상위 5개만

    async def _analyze_positive_feedback(
        self,
        feedback_list: List[FeedbackModel]
    ) -> List[Dict[str, Any]]:
        """긍정적 피드백 분석"""
        highlights = []

        try:
            # 높은 평점의 피드백 수집
            positive_feedback = [
                f for f in feedback_list
                if (f.rating and f.rating >= 4) or
                   (f.sentiment_score and f.sentiment_score in [SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE])
            ]

            # 긍정 유형별 분류
            positive_types = defaultdict(list)
            for feedback in positive_feedback:
                positive_types[feedback.feedback_type].append(feedback)

            # 상위 긍정사항 생성
            for feedback_type, feedback_items in positive_types.items():
                if len(feedback_items) >= 2:  # 2건 이상인 경우
                    highlight = {
                        'type': feedback_type.value,
                        'count': len(feedback_items),
                        'average_rating': statistics.mean([f.rating for f in feedback_items if f.rating]),
                        'sample_feedback': [
                            f.text_feedback for f in feedback_items[:3]
                            if f.text_feedback
                        ]
                    }
                    highlights.append(highlight)

            # 평점순 정렬
            highlights.sort(key=lambda x: x['average_rating'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to analyze positive feedback: {e}")

        return highlights[:5]  # 상위 5개만

    async def _predict_trends(
        self,
        feedback_list: List[FeedbackModel],
        stats: FeedbackStats
    ) -> Dict[str, Any]:
        """트렌드 예측 (간단한 구현)"""
        predictions = {}

        try:
            # 시간별 피드백 볼륨 트렌드
            daily_counts = defaultdict(int)
            for feedback in feedback_list:
                date_key = feedback.timestamp.strftime('%Y-%m-%d')
                daily_counts[date_key] += 1

            if len(daily_counts) >= 7:  # 최소 7일 데이터
                recent_avg = statistics.mean(list(daily_counts.values())[-7:])
                earlier_avg = statistics.mean(list(daily_counts.values())[:-7])

                if recent_avg > earlier_avg * 1.2:
                    predictions['volume_trend'] = "increasing"
                    predictions['volume_change'] = (recent_avg - earlier_avg) / earlier_avg
                elif recent_avg < earlier_avg * 0.8:
                    predictions['volume_trend'] = "decreasing"
                    predictions['volume_change'] = (recent_avg - earlier_avg) / earlier_avg
                else:
                    predictions['volume_trend'] = "stable"

            # 평점 트렌드
            if len(feedback_list) >= 20:
                recent_ratings = [f.rating for f in feedback_list[-20:] if f.rating]
                earlier_ratings = [f.rating for f in feedback_list[:-20] if f.rating]

                if recent_ratings and earlier_ratings:
                    recent_avg_rating = statistics.mean(recent_ratings)
                    earlier_avg_rating = statistics.mean(earlier_ratings)

                    if recent_avg_rating > earlier_avg_rating + 0.5:
                        predictions['rating_trend'] = "improving"
                    elif recent_avg_rating < earlier_avg_rating - 0.5:
                        predictions['rating_trend'] = "declining"
                    else:
                        predictions['rating_trend'] = "stable"

        except Exception as e:
            logger.error(f"Failed to predict trends: {e}")

        return predictions

    async def _recommend_actions(
        self,
        feedback_list: List[FeedbackModel],
        stats: FeedbackStats,
        critical_issues: List[str]
    ) -> List[Dict[str, str]]:
        """권장 액션 생성"""
        actions = []

        try:
            # 중요한 이슈 기반 권장사항
            if critical_issues:
                actions.append({
                    'priority': 'high',
                    'action': '중요한 이슈들을 우선적으로 해결하세요',
                    'details': f"{len(critical_issues)}개의 중요 이슈가 식별되었습니다"
                })

            # 평점 기반 권장사항
            if stats.average_rating and stats.average_rating < 3.0:
                actions.append({
                    'priority': 'high',
                    'action': '사용자 만족도 개선 프로그램을 시작하세요',
                    'details': f"현재 평균 평점: {stats.average_rating:.1f}"
                })

            # 해결률 기반 권장사항
            pending_count = stats.feedback_by_status.get('pending', 0)
            if pending_count > stats.total_feedback * 0.3:
                actions.append({
                    'priority': 'medium',
                    'action': '미해결 피드백 처리 시간을 단축하세요',
                    'details': f"{pending_count}개의 미해결 피드백이 있습니다"
                })

            # 도구 성능 기반 권장사항
            tool_issues = [f for f in feedback_list
                          if f.feedback_type == FeedbackType.TOOL_EXECUTION and f.rating and f.rating <= 2]
            if len(tool_issues) > 5:
                actions.append({
                    'priority': 'medium',
                    'action': '도구 안정성 개선에 집중하세요',
                    'details': f"{len(tool_issues)}건의 도구 실행 문제가 보고되었습니다"
                })

        except Exception as e:
            logger.error(f"Failed to recommend actions: {e}")

        return actions

    async def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """일일 피드백 리포트 생성"""
        if not date:
            date = datetime.now().date()

        start_date = datetime.combine(date, datetime.min.time())
        end_date = start_date + timedelta(days=1)

        try:
            analysis = await self.analyze_feedback(start_date, end_date, include_predictions=False)

            report = {
                'date': date.isoformat(),
                'summary': {
                    'total_feedback': analysis.stats.total_feedback,
                    'average_rating': analysis.stats.average_rating,
                    'critical_issues_count': len(analysis.critical_issues),
                    'resolution_rate': self._calculate_resolution_rate(analysis.stats)
                },
                'key_insights': analysis.key_insights[:3],  # 상위 3개
                'critical_issues': analysis.critical_issues,
                'top_complaint': analysis.common_complaints[0] if analysis.common_complaints else None,
                'top_highlight': analysis.positive_highlights[0] if analysis.positive_highlights else None
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return {}

    def _calculate_resolution_rate(self, stats: FeedbackStats) -> float:
        """해결률 계산"""
        resolved = stats.feedback_by_status.get('resolved', 0)
        closed = stats.feedback_by_status.get('closed', 0)
        total = stats.total_feedback

        if total == 0:
            return 0.0

        return (resolved + closed) / total
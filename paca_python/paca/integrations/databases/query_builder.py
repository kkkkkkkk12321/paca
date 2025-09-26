"""
Query Builder
SQL 쿼리를 동적으로 생성하는 빌더
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from ...core.utils.logger import PacaLogger


class JoinType(Enum):
    """조인 타입"""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"


class OrderDirection(Enum):
    """정렬 방향"""
    ASC = "ASC"
    DESC = "DESC"


@dataclass
class WhereCondition:
    """WHERE 조건"""
    column: str
    operator: str
    value: Any
    logical_operator: str = "AND"  # AND, OR


@dataclass
class JoinCondition:
    """JOIN 조건"""
    join_type: JoinType
    table: str
    on_condition: str
    alias: Optional[str] = None


class QueryBuilder:
    """SQL 쿼리 빌더"""

    def __init__(self):
        self.logger = PacaLogger("QueryBuilder")
        self.reset()

    def reset(self) -> 'QueryBuilder':
        """빌더 초기화"""
        self._select_fields: List[str] = []
        self._from_table: Optional[str] = None
        self._table_alias: Optional[str] = None
        self._joins: List[JoinCondition] = []
        self._where_conditions: List[WhereCondition] = []
        self._group_by: List[str] = []
        self._having_conditions: List[WhereCondition] = []
        self._order_by: List[Tuple[str, OrderDirection]] = []
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None

        # INSERT/UPDATE/DELETE용
        self._insert_table: Optional[str] = None
        self._insert_data: Dict[str, Any] = {}
        self._update_table: Optional[str] = None
        self._update_data: Dict[str, Any] = {}
        self._delete_table: Optional[str] = None

        return self

    def select(self, *fields: str) -> 'QueryBuilder':
        """SELECT 필드 지정"""
        if not fields:
            self._select_fields = ["*"]
        else:
            self._select_fields.extend(fields)
        return self

    def from_table(self, table: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """FROM 테이블 지정"""
        self._from_table = table
        self._table_alias = alias
        return self

    def join(
        self,
        table: str,
        on_condition: str,
        join_type: JoinType = JoinType.INNER,
        alias: Optional[str] = None
    ) -> 'QueryBuilder':
        """JOIN 추가"""
        join_condition = JoinCondition(
            join_type=join_type,
            table=table,
            on_condition=on_condition,
            alias=alias
        )
        self._joins.append(join_condition)
        return self

    def inner_join(self, table: str, on_condition: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """INNER JOIN"""
        return self.join(table, on_condition, JoinType.INNER, alias)

    def left_join(self, table: str, on_condition: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """LEFT JOIN"""
        return self.join(table, on_condition, JoinType.LEFT, alias)

    def right_join(self, table: str, on_condition: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """RIGHT JOIN"""
        return self.join(table, on_condition, JoinType.RIGHT, alias)

    def where(
        self,
        column: str,
        operator: str,
        value: Any,
        logical_operator: str = "AND"
    ) -> 'QueryBuilder':
        """WHERE 조건 추가"""
        condition = WhereCondition(
            column=column,
            operator=operator,
            value=value,
            logical_operator=logical_operator
        )
        self._where_conditions.append(condition)
        return self

    def where_equals(self, column: str, value: Any) -> 'QueryBuilder':
        """WHERE 등등 조건"""
        return self.where(column, "=", value)

    def where_not_equals(self, column: str, value: Any) -> 'QueryBuilder':
        """WHERE 부등 조건"""
        return self.where(column, "!=", value)

    def where_in(self, column: str, values: List[Any]) -> 'QueryBuilder':
        """WHERE IN 조건"""
        return self.where(column, "IN", values)

    def where_not_in(self, column: str, values: List[Any]) -> 'QueryBuilder':
        """WHERE NOT IN 조건"""
        return self.where(column, "NOT IN", values)

    def where_like(self, column: str, pattern: str) -> 'QueryBuilder':
        """WHERE LIKE 조건"""
        return self.where(column, "LIKE", pattern)

    def where_between(self, column: str, start: Any, end: Any) -> 'QueryBuilder':
        """WHERE BETWEEN 조건"""
        return self.where(column, "BETWEEN", f"{start} AND {end}")

    def where_null(self, column: str) -> 'QueryBuilder':
        """WHERE IS NULL 조건"""
        return self.where(column, "IS", "NULL")

    def where_not_null(self, column: str) -> 'QueryBuilder':
        """WHERE IS NOT NULL 조건"""
        return self.where(column, "IS NOT", "NULL")

    def or_where(self, column: str, operator: str, value: Any) -> 'QueryBuilder':
        """OR WHERE 조건"""
        return self.where(column, operator, value, "OR")

    def group_by(self, *columns: str) -> 'QueryBuilder':
        """GROUP BY 추가"""
        self._group_by.extend(columns)
        return self

    def having(
        self,
        column: str,
        operator: str,
        value: Any,
        logical_operator: str = "AND"
    ) -> 'QueryBuilder':
        """HAVING 조건 추가"""
        condition = WhereCondition(
            column=column,
            operator=operator,
            value=value,
            logical_operator=logical_operator
        )
        self._having_conditions.append(condition)
        return self

    def order_by(self, column: str, direction: OrderDirection = OrderDirection.ASC) -> 'QueryBuilder':
        """ORDER BY 추가"""
        self._order_by.append((column, direction))
        return self

    def order_by_asc(self, column: str) -> 'QueryBuilder':
        """ORDER BY ASC"""
        return self.order_by(column, OrderDirection.ASC)

    def order_by_desc(self, column: str) -> 'QueryBuilder':
        """ORDER BY DESC"""
        return self.order_by(column, OrderDirection.DESC)

    def limit(self, count: int) -> 'QueryBuilder':
        """LIMIT 설정"""
        self._limit_value = count
        return self

    def offset(self, count: int) -> 'QueryBuilder':
        """OFFSET 설정"""
        self._offset_value = count
        return self

    def paginate(self, page: int, per_page: int) -> 'QueryBuilder':
        """페이지네이션"""
        offset = (page - 1) * per_page
        return self.limit(per_page).offset(offset)

    def insert_into(self, table: str) -> 'QueryBuilder':
        """INSERT INTO 설정"""
        self._insert_table = table
        return self

    def values(self, **data: Any) -> 'QueryBuilder':
        """INSERT VALUES 설정"""
        self._insert_data.update(data)
        return self

    def update_table(self, table: str) -> 'QueryBuilder':
        """UPDATE 테이블 설정"""
        self._update_table = table
        return self

    def set_values(self, **data: Any) -> 'QueryBuilder':
        """UPDATE SET 값 설정"""
        self._update_data.update(data)
        return self

    def delete_from(self, table: str) -> 'QueryBuilder':
        """DELETE FROM 설정"""
        self._delete_table = table
        return self

    def build_select(self) -> Tuple[str, List[Any]]:
        """SELECT 쿼리 빌드"""
        if not self._from_table:
            raise ValueError("FROM table is required for SELECT query")

        # SELECT 절
        select_clause = "SELECT " + ", ".join(self._select_fields)

        # FROM 절
        from_clause = f"FROM {self._from_table}"
        if self._table_alias:
            from_clause += f" AS {self._table_alias}"

        query_parts = [select_clause, from_clause]
        params = []

        # JOIN 절
        for join in self._joins:
            join_clause = f"{join.join_type.value} {join.table}"
            if join.alias:
                join_clause += f" AS {join.alias}"
            join_clause += f" ON {join.on_condition}"
            query_parts.append(join_clause)

        # WHERE 절
        if self._where_conditions:
            where_clause, where_params = self._build_where_clause(self._where_conditions)
            query_parts.append(where_clause)
            params.extend(where_params)

        # GROUP BY 절
        if self._group_by:
            query_parts.append("GROUP BY " + ", ".join(self._group_by))

        # HAVING 절
        if self._having_conditions:
            having_clause, having_params = self._build_where_clause(self._having_conditions, "HAVING")
            query_parts.append(having_clause)
            params.extend(having_params)

        # ORDER BY 절
        if self._order_by:
            order_items = [f"{col} {direction.value}" for col, direction in self._order_by]
            query_parts.append("ORDER BY " + ", ".join(order_items))

        # LIMIT 절
        if self._limit_value is not None:
            query_parts.append(f"LIMIT {self._limit_value}")

        # OFFSET 절
        if self._offset_value is not None:
            query_parts.append(f"OFFSET {self._offset_value}")

        query = " ".join(query_parts)
        return query, params

    def build_insert(self) -> Tuple[str, List[Any]]:
        """INSERT 쿼리 빌드"""
        if not self._insert_table:
            raise ValueError("INSERT table is required")

        if not self._insert_data:
            raise ValueError("INSERT data is required")

        columns = list(self._insert_data.keys())
        values = list(self._insert_data.values())
        placeholders = ["?" for _ in values]

        query = f"INSERT INTO {self._insert_table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        return query, values

    def build_update(self) -> Tuple[str, List[Any]]:
        """UPDATE 쿼리 빌드"""
        if not self._update_table:
            raise ValueError("UPDATE table is required")

        if not self._update_data:
            raise ValueError("UPDATE data is required")

        # SET 절
        set_items = [f"{col} = ?" for col in self._update_data.keys()]
        set_clause = "SET " + ", ".join(set_items)

        query_parts = [f"UPDATE {self._update_table}", set_clause]
        params = list(self._update_data.values())

        # WHERE 절
        if self._where_conditions:
            where_clause, where_params = self._build_where_clause(self._where_conditions)
            query_parts.append(where_clause)
            params.extend(where_params)

        query = " ".join(query_parts)
        return query, params

    def build_delete(self) -> Tuple[str, List[Any]]:
        """DELETE 쿼리 빌드"""
        if not self._delete_table:
            raise ValueError("DELETE table is required")

        query_parts = [f"DELETE FROM {self._delete_table}"]
        params = []

        # WHERE 절
        if self._where_conditions:
            where_clause, where_params = self._build_where_clause(self._where_conditions)
            query_parts.append(where_clause)
            params.extend(where_params)

        query = " ".join(query_parts)
        return query, params

    def _build_where_clause(
        self,
        conditions: List[WhereCondition],
        clause_type: str = "WHERE"
    ) -> Tuple[str, List[Any]]:
        """WHERE/HAVING 절 빌드"""
        if not conditions:
            return "", []

        condition_parts = []
        params = []

        for i, condition in enumerate(conditions):
            # 첫 번째 조건이 아니면 논리 연산자 추가
            if i > 0:
                condition_parts.append(condition.logical_operator)

            # 조건 생성
            if condition.operator.upper() in ["IN", "NOT IN"]:
                if isinstance(condition.value, (list, tuple)):
                    placeholders = ", ".join(["?" for _ in condition.value])
                    condition_parts.append(f"{condition.column} {condition.operator} ({placeholders})")
                    params.extend(condition.value)
                else:
                    condition_parts.append(f"{condition.column} {condition.operator} (?)")
                    params.append(condition.value)
            elif condition.operator.upper() in ["IS", "IS NOT"] and str(condition.value).upper() == "NULL":
                condition_parts.append(f"{condition.column} {condition.operator} NULL")
            else:
                condition_parts.append(f"{condition.column} {condition.operator} ?")
                params.append(condition.value)

        clause = f"{clause_type} " + " ".join(condition_parts)
        return clause, params

    def to_sql(self) -> Tuple[str, List[Any]]:
        """SQL 쿼리 생성 (자동 감지)"""
        if self._insert_table:
            return self.build_insert()
        elif self._update_table:
            return self.build_update()
        elif self._delete_table:
            return self.build_delete()
        else:
            return self.build_select()

    def get_count_query(self) -> Tuple[str, List[Any]]:
        """COUNT 쿼리 생성"""
        original_select = self._select_fields.copy()
        original_order = self._order_by.copy()
        original_limit = self._limit_value
        original_offset = self._offset_value

        # COUNT용으로 수정
        self._select_fields = ["COUNT(*) as total_count"]
        self._order_by = []
        self._limit_value = None
        self._offset_value = None

        query, params = self.build_select()

        # 원래 값으로 복원
        self._select_fields = original_select
        self._order_by = original_order
        self._limit_value = original_limit
        self._offset_value = original_offset

        return query, params

    def clone(self) -> 'QueryBuilder':
        """쿼리 빌더 복사"""
        new_builder = QueryBuilder()
        new_builder._select_fields = self._select_fields.copy()
        new_builder._from_table = self._from_table
        new_builder._table_alias = self._table_alias
        new_builder._joins = self._joins.copy()
        new_builder._where_conditions = self._where_conditions.copy()
        new_builder._group_by = self._group_by.copy()
        new_builder._having_conditions = self._having_conditions.copy()
        new_builder._order_by = self._order_by.copy()
        new_builder._limit_value = self._limit_value
        new_builder._offset_value = self._offset_value
        new_builder._insert_table = self._insert_table
        new_builder._insert_data = self._insert_data.copy()
        new_builder._update_table = self._update_table
        new_builder._update_data = self._update_data.copy()
        new_builder._delete_table = self._delete_table
        return new_builder


# 편의 함수들
def select(*fields: str) -> QueryBuilder:
    """SELECT 쿼리 빌더 시작"""
    return QueryBuilder().select(*fields)


def insert_into(table: str) -> QueryBuilder:
    """INSERT 쿼리 빌더 시작"""
    return QueryBuilder().insert_into(table)


def update_table(table: str) -> QueryBuilder:
    """UPDATE 쿼리 빌더 시작"""
    return QueryBuilder().update_table(table)


def delete_from(table: str) -> QueryBuilder:
    """DELETE 쿼리 빌더 시작"""
    return QueryBuilder().delete_from(table)


# 사용 예시들
def create_example_queries():
    """예시 쿼리들"""

    # SELECT 예시
    select_query, params = (
        select("id", "name", "email")
        .from_table("users", "u")
        .left_join("profiles", "u.id = p.user_id", "p")
        .where_equals("u.active", True)
        .where_like("u.name", "%admin%")
        .order_by_desc("u.created_at")
        .limit(10)
        .to_sql()
    )

    # INSERT 예시
    insert_query, params = (
        insert_into("users")
        .values(name="John Doe", email="john@example.com", active=True)
        .to_sql()
    )

    # UPDATE 예시
    update_query, params = (
        update_table("users")
        .set_values(last_login="2024-01-01", active=True)
        .where_equals("id", 123)
        .to_sql()
    )

    # DELETE 예시
    delete_query, params = (
        delete_from("users")
        .where_equals("active", False)
        .where("created_at", "<", "2023-01-01")
        .to_sql()
    )

    return [
        (select_query, params),
        (insert_query, params),
        (update_query, params),
        (delete_query, params)
    ]
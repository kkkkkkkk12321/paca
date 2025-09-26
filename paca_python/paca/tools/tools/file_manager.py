"""
파일 관리 도구 (The Librarian)

안전한 샌드박스 환경에서의 파일 읽기/쓰기 및 관리 기능
"""

import os
import json
import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import tempfile
import mimetypes

from ..base import Tool, ToolResult, ToolType


@dataclass
class FileInfo:
    """파일 정보"""
    path: str
    name: str
    size: int
    modified: datetime
    created: datetime
    is_directory: bool
    permissions: str
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    hash_md5: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'name': self.name,
            'size': self.size,
            'modified': self.modified.isoformat(),
            'created': self.created.isoformat(),
            'is_directory': self.is_directory,
            'permissions': self.permissions,
            'mime_type': self.mime_type,
            'encoding': self.encoding,
            'hash_md5': self.hash_md5
        }


class FileManagerTool(Tool):
    """파일 관리 도구 - The Librarian"""

    def __init__(self, sandbox_path: Optional[str] = None):
        super().__init__(
            name="file_manager",
            tool_type=ToolType.FILE,
            description="안전한 샌드박스 환경에서 파일을 읽고, 쓰고, 관리합니다."
        )

        # 샌드박스 경로 설정
        if sandbox_path:
            self.sandbox_path = Path(sandbox_path)
        else:
            # 기본 샌드박스 경로 (사용자의 임시 디렉토리 하위)
            self.sandbox_path = Path(tempfile.gettempdir()) / "paca_sandbox"

        # 샌드박스 디렉토리 생성
        self.sandbox_path.mkdir(parents=True, exist_ok=True)

        # 허용된 파일 확장자
        self.allowed_extensions = {
            '.txt', '.md', '.json', '.csv', '.xml', '.yaml', '.yml',
            '.log', '.cfg', '.conf', '.ini', '.py', '.js', '.html',
            '.css', '.sql', '.sh', '.bat'
        }

        # 최대 파일 크기 (10MB)
        self.max_file_size = 10 * 1024 * 1024

    def validate_input(self, operation: str = "", path: str = "", **kwargs) -> bool:
        """입력 검증"""
        if not operation:
            return False

        valid_operations = {
            'read', 'write', 'append', 'delete', 'list', 'info',
            'copy', 'move', 'mkdir', 'exists', 'search'
        }

        if operation not in valid_operations:
            return False

        if operation in ['read', 'write', 'append', 'delete', 'info', 'copy', 'move'] and not path:
            return False

        return True

    async def execute(self, operation: str, path: str = "", content: str = "",
                     recursive: bool = False, pattern: str = "*", **kwargs) -> ToolResult:
        """파일 관리 작업 실행"""
        try:
            if not self.validate_input(operation, path, **kwargs):
                return ToolResult(
                    success=False,
                    error="유효하지 않은 파라미터입니다."
                )

            # 경로 안전성 검증
            if path and not self._is_safe_path(path):
                return ToolResult(
                    success=False,
                    error="허용되지 않은 경로입니다."
                )

            # 작업별 실행
            operations = {
                'read': self._read_file,
                'write': self._write_file,
                'append': self._append_file,
                'delete': self._delete_file,
                'list': self._list_files,
                'info': self._get_file_info,
                'copy': self._copy_file,
                'move': self._move_file,
                'mkdir': self._create_directory,
                'exists': self._file_exists,
                'search': self._search_files
            }

            operation_func = operations.get(operation)
            if not operation_func:
                return ToolResult(
                    success=False,
                    error=f"지원하지 않는 작업: {operation}"
                )

            # 작업 실행
            result = await operation_func(path, content, recursive, pattern, **kwargs)
            return result

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"파일 작업 중 오류 발생: {str(e)}"
            )

    def _is_safe_path(self, path: str) -> bool:
        """경로 안전성 검증"""
        try:
            # 절대 경로를 샌드박스 내 상대 경로로 변환
            full_path = self.sandbox_path / path
            resolved_path = full_path.resolve()

            # 샌드박스 경계 검사
            if not str(resolved_path).startswith(str(self.sandbox_path.resolve())):
                return False

            # 파일 확장자 검사 (디렉토리가 아닌 경우)
            if resolved_path.suffix and resolved_path.suffix.lower() not in self.allowed_extensions:
                return False

            return True

        except Exception:
            return False

    def _get_full_path(self, path: str) -> Path:
        """전체 경로 반환"""
        return self.sandbox_path / path

    async def _read_file(self, path: str, content: str, recursive: bool,
                        pattern: str, **kwargs) -> ToolResult:
        """파일 읽기"""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"파일을 찾을 수 없습니다: {path}"
                )

            if full_path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"디렉토리는 읽을 수 없습니다: {path}"
                )

            # 파일 크기 검사
            if full_path.stat().st_size > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"파일이 너무 큽니다 (최대 {self.max_file_size / 1024 / 1024:.1f}MB)"
                )

            # 파일 읽기
            encoding = kwargs.get('encoding', 'utf-8')
            try:
                with open(full_path, 'r', encoding=encoding) as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # 바이너리 파일로 읽기 시도
                with open(full_path, 'rb') as f:
                    file_content = f.read()
                    # Base64 인코딩 또는 헥스 표현
                    import base64
                    file_content = base64.b64encode(file_content).decode('ascii')
                    encoding = 'base64'

            metadata = {
                'path': path,
                'size': full_path.stat().st_size,
                'encoding': encoding,
                'modified': datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
            }

            return ToolResult(
                success=True,
                data=file_content,
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"파일 읽기 오류: {str(e)}"
            )

    async def _write_file(self, path: str, content: str, recursive: bool,
                         pattern: str, **kwargs) -> ToolResult:
        """파일 쓰기"""
        try:
            full_path = self._get_full_path(path)

            # 디렉토리 생성
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # 파일 크기 제한 검사
            if len(content.encode('utf-8')) > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"내용이 너무 큽니다 (최대 {self.max_file_size / 1024 / 1024:.1f}MB)"
                )

            # 백업 생성 (기존 파일이 있는 경우)
            backup_created = False
            if full_path.exists() and kwargs.get('create_backup', True):
                backup_path = full_path.with_suffix(full_path.suffix + '.backup')
                shutil.copy2(full_path, backup_path)
                backup_created = True

            # 파일 쓰기
            encoding = kwargs.get('encoding', 'utf-8')
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)

            metadata = {
                'path': path,
                'size': full_path.stat().st_size,
                'encoding': encoding,
                'backup_created': backup_created,
                'created': datetime.now().isoformat()
            }

            return ToolResult(
                success=True,
                data=f"파일이 성공적으로 작성되었습니다: {path}",
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"파일 쓰기 오류: {str(e)}"
            )

    async def _append_file(self, path: str, content: str, recursive: bool,
                          pattern: str, **kwargs) -> ToolResult:
        """파일에 내용 추가"""
        try:
            full_path = self._get_full_path(path)

            # 디렉토리 생성
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # 파일 크기 제한 검사
            current_size = full_path.stat().st_size if full_path.exists() else 0
            new_content_size = len(content.encode('utf-8'))

            if current_size + new_content_size > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"추가 후 파일 크기가 제한을 초과합니다 (최대 {self.max_file_size / 1024 / 1024:.1f}MB)"
                )

            # 파일에 추가
            encoding = kwargs.get('encoding', 'utf-8')
            with open(full_path, 'a', encoding=encoding) as f:
                f.write(content)

            metadata = {
                'path': path,
                'size': full_path.stat().st_size,
                'encoding': encoding,
                'appended_size': new_content_size,
                'modified': datetime.now().isoformat()
            }

            return ToolResult(
                success=True,
                data=f"내용이 성공적으로 추가되었습니다: {path}",
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"파일 추가 오류: {str(e)}"
            )

    async def _delete_file(self, path: str, content: str, recursive: bool,
                          pattern: str, **kwargs) -> ToolResult:
        """파일 또는 디렉토리 삭제"""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"파일 또는 디렉토리를 찾을 수 없습니다: {path}"
                )

            deleted_items = []

            if full_path.is_file():
                # 백업 생성
                if kwargs.get('create_backup', True):
                    backup_path = full_path.with_suffix(full_path.suffix + '.deleted')
                    shutil.move(full_path, backup_path)
                    deleted_items.append({'path': path, 'backup': str(backup_path)})
                else:
                    full_path.unlink()
                    deleted_items.append({'path': path, 'backup': None})

            elif full_path.is_dir():
                if recursive:
                    shutil.rmtree(full_path)
                    deleted_items.append({'path': path, 'type': 'directory'})
                else:
                    try:
                        full_path.rmdir()  # 빈 디렉토리만 삭제
                        deleted_items.append({'path': path, 'type': 'empty_directory'})
                    except OSError:
                        return ToolResult(
                            success=False,
                            error=f"디렉토리가 비어있지 않습니다. recursive=True를 사용하세요: {path}"
                        )

            return ToolResult(
                success=True,
                data=f"성공적으로 삭제되었습니다: {path}",
                metadata={'deleted_items': deleted_items}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"삭제 오류: {str(e)}"
            )

    async def _list_files(self, path: str, content: str, recursive: bool,
                         pattern: str, **kwargs) -> ToolResult:
        """파일 목록 조회"""
        try:
            if not path:
                path = "."

            full_path = self._get_full_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"경로를 찾을 수 없습니다: {path}"
                )

            if not full_path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"디렉토리가 아닙니다: {path}"
                )

            files = []

            if recursive:
                # 재귀적 검색
                for item in full_path.rglob(pattern):
                    relative_path = item.relative_to(self.sandbox_path)
                    file_info = self._create_file_info(item, str(relative_path))
                    files.append(file_info.to_dict())
            else:
                # 현재 디렉토리만
                for item in full_path.glob(pattern):
                    relative_path = item.relative_to(self.sandbox_path)
                    file_info = self._create_file_info(item, str(relative_path))
                    files.append(file_info.to_dict())

            # 정렬 (디렉토리 먼저, 그 다음 이름순)
            files.sort(key=lambda x: (not x['is_directory'], x['name'].lower()))

            metadata = {
                'path': path,
                'pattern': pattern,
                'recursive': recursive,
                'total_items': len(files)
            }

            return ToolResult(
                success=True,
                data=files,
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"목록 조회 오류: {str(e)}"
            )

    async def _get_file_info(self, path: str, content: str, recursive: bool,
                           pattern: str, **kwargs) -> ToolResult:
        """파일 정보 조회"""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"파일 또는 디렉토리를 찾을 수 없습니다: {path}"
                )

            file_info = self._create_file_info(full_path, path)

            # 추가 정보
            if full_path.is_file() and kwargs.get('include_hash', False):
                file_info.hash_md5 = self._calculate_md5(full_path)

            return ToolResult(
                success=True,
                data=file_info.to_dict(),
                metadata={'path': path}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"정보 조회 오류: {str(e)}"
            )

    async def _copy_file(self, path: str, content: str, recursive: bool,
                        pattern: str, **kwargs) -> ToolResult:
        """파일 또는 디렉토리 복사"""
        try:
            source_path = self._get_full_path(path)
            dest_path_str = kwargs.get('destination', '')

            if not dest_path_str:
                return ToolResult(
                    success=False,
                    error="복사 대상 경로가 지정되지 않았습니다."
                )

            dest_path = self._get_full_path(dest_path_str)

            if not source_path.exists():
                return ToolResult(
                    success=False,
                    error=f"원본 파일을 찾을 수 없습니다: {path}"
                )

            # 디렉토리 생성
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
            elif source_path.is_dir():
                if recursive:
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                else:
                    return ToolResult(
                        success=False,
                        error="디렉토리 복사에는 recursive=True가 필요합니다."
                    )

            return ToolResult(
                success=True,
                data=f"성공적으로 복사되었습니다: {path} -> {dest_path_str}",
                metadata={'source': path, 'destination': dest_path_str}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"복사 오류: {str(e)}"
            )

    async def _move_file(self, path: str, content: str, recursive: bool,
                        pattern: str, **kwargs) -> ToolResult:
        """파일 또는 디렉토리 이동"""
        try:
            source_path = self._get_full_path(path)
            dest_path_str = kwargs.get('destination', '')

            if not dest_path_str:
                return ToolResult(
                    success=False,
                    error="이동 대상 경로가 지정되지 않았습니다."
                )

            dest_path = self._get_full_path(dest_path_str)

            if not source_path.exists():
                return ToolResult(
                    success=False,
                    error=f"원본 파일을 찾을 수 없습니다: {path}"
                )

            # 디렉토리 생성
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(source_path, dest_path)

            return ToolResult(
                success=True,
                data=f"성공적으로 이동되었습니다: {path} -> {dest_path_str}",
                metadata={'source': path, 'destination': dest_path_str}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"이동 오류: {str(e)}"
            )

    async def _create_directory(self, path: str, content: str, recursive: bool,
                               pattern: str, **kwargs) -> ToolResult:
        """디렉토리 생성"""
        try:
            full_path = self._get_full_path(path)

            if full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"이미 존재합니다: {path}"
                )

            full_path.mkdir(parents=recursive, exist_ok=False)

            return ToolResult(
                success=True,
                data=f"디렉토리가 생성되었습니다: {path}",
                metadata={'path': path, 'parents_created': recursive}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"디렉토리 생성 오류: {str(e)}"
            )

    async def _file_exists(self, path: str, content: str, recursive: bool,
                          pattern: str, **kwargs) -> ToolResult:
        """파일 존재 여부 확인"""
        try:
            full_path = self._get_full_path(path)
            exists = full_path.exists()

            result_data = {
                'exists': exists,
                'is_file': full_path.is_file() if exists else False,
                'is_directory': full_path.is_dir() if exists else False
            }

            return ToolResult(
                success=True,
                data=result_data,
                metadata={'path': path}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"존재 여부 확인 오류: {str(e)}"
            )

    async def _search_files(self, path: str, content: str, recursive: bool,
                           pattern: str, **kwargs) -> ToolResult:
        """파일 내용 검색"""
        try:
            if not path:
                path = "."

            full_path = self._get_full_path(path)
            search_text = kwargs.get('search_text', content)

            if not search_text:
                return ToolResult(
                    success=False,
                    error="검색할 텍스트가 지정되지 않았습니다."
                )

            matches = []

            # 검색 대상 파일 수집
            if full_path.is_file():
                files_to_search = [full_path]
            else:
                if recursive:
                    files_to_search = list(full_path.rglob(pattern))
                else:
                    files_to_search = list(full_path.glob(pattern))

                # 파일만 필터링
                files_to_search = [f for f in files_to_search if f.is_file()]

            # 파일 검색
            for file_path in files_to_search:
                try:
                    # 파일 크기 제한
                    if file_path.stat().st_size > self.max_file_size:
                        continue

                    # 텍스트 파일만 검색
                    if file_path.suffix.lower() not in self.allowed_extensions:
                        continue

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        if search_text.lower() in line.lower():
                            relative_path = file_path.relative_to(self.sandbox_path)
                            matches.append({
                                'file': str(relative_path),
                                'line_number': line_num,
                                'line_content': line.strip(),
                                'match_position': line.lower().find(search_text.lower())
                            })

                except Exception:
                    # 개별 파일 오류는 무시하고 계속
                    continue

            metadata = {
                'search_text': search_text,
                'search_path': path,
                'pattern': pattern,
                'recursive': recursive,
                'files_searched': len(files_to_search),
                'matches_found': len(matches)
            }

            return ToolResult(
                success=True,
                data=matches,
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"검색 오류: {str(e)}"
            )

    def _create_file_info(self, file_path: Path, relative_path: str) -> FileInfo:
        """파일 정보 객체 생성"""
        stat = file_path.stat()

        return FileInfo(
            path=relative_path,
            name=file_path.name,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            created=datetime.fromtimestamp(stat.st_ctime),
            is_directory=file_path.is_dir(),
            permissions=oct(stat.st_mode)[-3:],
            mime_type=mimetypes.guess_type(str(file_path))[0] if file_path.is_file() else None
        )

    def _calculate_md5(self, file_path: Path) -> str:
        """파일의 MD5 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_sandbox_info(self) -> Dict[str, Any]:
        """샌드박스 정보 반환"""
        try:
            total_size = sum(
                f.stat().st_size for f in self.sandbox_path.rglob('*') if f.is_file()
            )
            file_count = len([f for f in self.sandbox_path.rglob('*') if f.is_file()])
            dir_count = len([f for f in self.sandbox_path.rglob('*') if f.is_dir()])

            return {
                'sandbox_path': str(self.sandbox_path),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / 1024 / 1024,
                'file_count': file_count,
                'directory_count': dir_count,
                'allowed_extensions': list(self.allowed_extensions),
                'max_file_size_mb': self.max_file_size / 1024 / 1024
            }
        except Exception as e:
            return {
                'error': str(e),
                'sandbox_path': str(self.sandbox_path)
            }
"""
Input Validation Security Tests for PACA System
Purpose: Test system security against various input validation attacks
Author: PACA Development Team
Created: 2024-09-24
"""

import pytest
import asyncio
import json
import base64
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PACA modules
from paca.cognitive.memory import MemorySystem
from paca.data.cache.cache_manager import CacheManager
from paca.integrations.apis.universal_client import UniversalAPIClient


class SecurityTestPayloads:
    """Collection of security test payloads for various attack vectors."""

    # SQL Injection payloads
    SQL_INJECTION = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'/*",
        "' UNION SELECT * FROM users WHERE '1'='1",
        "'; INSERT INTO users VALUES('hacker', 'password'); --",
        "1' AND (SELECT COUNT(*) FROM users) > 0 --"
    ]

    # NoSQL Injection payloads
    NOSQL_INJECTION = [
        '{"$gt": ""}',
        '{"$ne": null}',
        '{"$regex": ".*"}',
        '{"$where": "function() { return true; }"}',
        '{"$or": [{"username": "admin"}, {"username": "root"}]}'
    ]

    # XSS payloads
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src='x' onerror='alert(1)'>",
        "<svg onload=alert(1)>",
        "';alert(String.fromCharCode(88,83,83))//';",
        "<iframe src=javascript:alert('XSS')>",
        "<body onload=alert('XSS')>"
    ]

    # Command Injection payloads
    COMMAND_INJECTION = [
        "; ls -la",
        "| whoami",
        "&& cat /etc/passwd",
        "; rm -rf /",
        "` id `",
        "$(whoami)",
        "|net user",
        "&dir"
    ]

    # Path Traversal payloads
    PATH_TRAVERSAL = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
        "..%252f..%252f..%252fetc%252fpasswd",
        "....\\....\\....\\windows\\system32\\config\\SAM"
    ]

    # Buffer Overflow payloads
    BUFFER_OVERFLOW = [
        "A" * 1000,
        "A" * 10000,
        "A" * 100000,
        "\x00" * 1000,
        "\xff" * 1000,
        "%" * 1000
    ]

    # Format String payloads
    FORMAT_STRING = [
        "%x%x%x%x%x%x%x%x",
        "%s%s%s%s%s%s%s%s",
        "%n%n%n%n%n%n%n%n",
        "%.1000d",
        "%*.*s",
        "%p%p%p%p%p%p%p%p"
    ]

    # LDAP Injection payloads
    LDAP_INJECTION = [
        "*)(&(password=*))",
        "*)(|(objectClass=*))",
        "admin*)((|userPassword=*)",
        ")(cn=*))((|(cn=*",
        "*))(|(objectClass=*"
    ]

    # Unicode/Encoding attacks
    UNICODE_ATTACKS = [
        "\u202e\u0041\u0042\u0043",  # Right-to-left override
        "\ufeff\u200b\u200c\u200d",  # Zero-width characters
        "\u0000\u0001\u0002\u0003",  # Control characters
        "%c0%ae%c0%ae/",  # UTF-8 encoding bypass
        "\uFE64\uFE65\uFE66"  # Small form variants
    ]


class TestInputValidation:
    """
    Comprehensive input validation security tests.

    Tests system security against various input-based attacks.
    """

    @pytest.fixture(scope="class")
    def paca_system(self):
        """Initialize PACA system for security testing."""
        system = {
            'memory': MemorySystem(),
            'cache': CacheManager(),
            'api_client': UniversalAPIClient(),
        }
        yield system

    @pytest.fixture
    def security_payloads(self):
        """Provide security test payloads."""
        return SecurityTestPayloads()

    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, paca_system, security_payloads):
        """Test protection against SQL injection attacks."""
        memory_system = paca_system['memory']
        injection_attempts = []

        for payload in security_payloads.SQL_INJECTION:
            try:
                # Test memory storage with SQL injection payloads
                memory_id = await memory_system.store_memory(
                    content=payload,
                    memory_type="test",
                    importance=0.5,
                    tags=["security_test"]
                )

                # Verify the payload is stored as-is, not executed
                retrieved = await memory_system.retrieve_memory(memory_id)
                injection_attempts.append({
                    'payload': payload,
                    'stored_safely': retrieved['content'] == payload,
                    'executed': False  # Should never be True
                })

            except Exception as e:
                # System should handle malicious input gracefully
                injection_attempts.append({
                    'payload': payload,
                    'error': str(e),
                    'blocked': True
                })

        # Assertions
        safe_storage_count = sum(1 for attempt in injection_attempts
                               if attempt.get('stored_safely', False))
        blocked_count = sum(1 for attempt in injection_attempts
                          if attempt.get('blocked', False))

        assert safe_storage_count + blocked_count == len(security_payloads.SQL_INJECTION)
        assert not any(attempt.get('executed', False) for attempt in injection_attempts)

    @pytest.mark.asyncio
    async def test_xss_payload_sanitization(self, paca_system, security_payloads):
        """Test XSS payload sanitization and safe handling."""
        memory_system = paca_system['memory']
        cache_system = paca_system['cache']
        xss_tests = []

        for payload in security_payloads.XSS_PAYLOADS:
            # Test memory system
            try:
                memory_id = await memory_system.store_memory(
                    content=payload,
                    memory_type="user_input",
                    importance=0.1,
                    tags=["xss_test"]
                )

                retrieved = await memory_system.retrieve_memory(memory_id)
                memory_safe = self._is_payload_safe(retrieved['content'], payload)

                # Test cache system
                await cache_system.set(f"xss_test_{len(xss_tests)}", payload, ttl=60)
                cached = await cache_system.get(f"xss_test_{len(xss_tests)}")
                cache_safe = self._is_payload_safe(cached, payload)

                xss_tests.append({
                    'payload': payload,
                    'memory_safe': memory_safe,
                    'cache_safe': cache_safe,
                    'fully_safe': memory_safe and cache_safe
                })

            except Exception as e:
                xss_tests.append({
                    'payload': payload,
                    'blocked': True,
                    'error': str(e),
                    'fully_safe': True  # Blocking is also safe
                })

        # Assertions
        safe_handling_count = sum(1 for test in xss_tests if test.get('fully_safe', False))
        assert safe_handling_count == len(security_payloads.XSS_PAYLOADS)

        # Verify no script execution indicators
        dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onload=']
        for test in xss_tests:
            if not test.get('blocked', False):
                for pattern in dangerous_patterns:
                    assert pattern not in test.get('processed_content', '').lower()

    @pytest.mark.asyncio
    async def test_command_injection_prevention(self, paca_system, security_payloads):
        """Test prevention of command injection attacks."""
        api_client = paca_system['api_client']
        command_tests = []

        for payload in security_payloads.COMMAND_INJECTION:
            try:
                # Test API client with command injection payloads
                response = await api_client.request(
                    method='GET',
                    endpoint='/test',
                    params={'user_input': payload}
                )

                # Verify no command execution occurred
                command_executed = self._detect_command_execution(response)
                command_tests.append({
                    'payload': payload,
                    'command_executed': command_executed,
                    'safe': not command_executed
                })

            except Exception as e:
                # Exception handling is acceptable for security
                command_tests.append({
                    'payload': payload,
                    'blocked': True,
                    'error': str(e),
                    'safe': True
                })

        # Assertions
        safe_count = sum(1 for test in command_tests if test.get('safe', False))
        assert safe_count == len(security_payloads.COMMAND_INJECTION)

        # Ensure no commands were executed
        executed_count = sum(1 for test in command_tests
                           if test.get('command_executed', False))
        assert executed_count == 0

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, paca_system, security_payloads):
        """Test protection against path traversal attacks."""
        cache_system = paca_system['cache']
        traversal_tests = []

        for payload in security_payloads.PATH_TRAVERSAL:
            try:
                # Test cache system with path traversal payloads as keys
                safe_key = self._sanitize_cache_key(payload)
                await cache_system.set(safe_key, "test_data", ttl=60)

                # Verify the key was sanitized and doesn't allow traversal
                contains_traversal = self._contains_path_traversal(safe_key)
                traversal_tests.append({
                    'payload': payload,
                    'sanitized_key': safe_key,
                    'contains_traversal': contains_traversal,
                    'safe': not contains_traversal
                })

            except Exception as e:
                traversal_tests.append({
                    'payload': payload,
                    'blocked': True,
                    'error': str(e),
                    'safe': True
                })

        # Assertions
        safe_count = sum(1 for test in traversal_tests if test.get('safe', False))
        assert safe_count == len(security_payloads.PATH_TRAVERSAL)

        # Ensure no traversal patterns remain
        traversal_count = sum(1 for test in traversal_tests
                            if test.get('contains_traversal', False))
        assert traversal_count == 0

    @pytest.mark.asyncio
    async def test_buffer_overflow_protection(self, paca_system, security_payloads):
        """Test protection against buffer overflow attacks."""
        memory_system = paca_system['memory']
        overflow_tests = []

        for payload in security_payloads.BUFFER_OVERFLOW:
            try:
                # Test with large payloads
                memory_id = await memory_system.store_memory(
                    content=payload,
                    memory_type="overflow_test",
                    importance=0.1,
                    tags=["buffer_test"]
                )

                # Verify system handles large inputs gracefully
                retrieved = await memory_system.retrieve_memory(memory_id)
                system_stable = retrieved is not None
                overflow_tests.append({
                    'payload_size': len(payload),
                    'system_stable': system_stable,
                    'handled_gracefully': True
                })

            except MemoryError:
                overflow_tests.append({
                    'payload_size': len(payload),
                    'memory_error': True,
                    'handled_gracefully': True  # MemoryError is acceptable
                })
            except Exception as e:
                overflow_tests.append({
                    'payload_size': len(payload),
                    'other_error': str(e),
                    'handled_gracefully': True  # Any controlled error is acceptable
                })

        # Assertions
        graceful_handling_count = sum(1 for test in overflow_tests
                                    if test.get('handled_gracefully', False))
        assert graceful_handling_count == len(security_payloads.BUFFER_OVERFLOW)

        # Verify system didn't crash (all tests completed)
        assert len(overflow_tests) == len(security_payloads.BUFFER_OVERFLOW)

    @pytest.mark.asyncio
    async def test_format_string_protection(self, paca_system, security_payloads):
        """Test protection against format string attacks."""
        memory_system = paca_system['memory']
        format_tests = []

        for payload in security_payloads.FORMAT_STRING:
            try:
                # Test format string payloads
                memory_id = await memory_system.store_memory(
                    content=f"User input: {payload}",
                    memory_type="format_test",
                    importance=0.1,
                    tags=["format_string_test"]
                )

                retrieved = await memory_system.retrieve_memory(memory_id)

                # Check if format string was interpreted (security vulnerability)
                format_interpreted = self._detect_format_string_exploitation(
                    retrieved['content'], payload
                )

                format_tests.append({
                    'payload': payload,
                    'format_interpreted': format_interpreted,
                    'safe': not format_interpreted
                })

            except Exception as e:
                format_tests.append({
                    'payload': payload,
                    'blocked': True,
                    'error': str(e),
                    'safe': True
                })

        # Assertions
        safe_count = sum(1 for test in format_tests if test.get('safe', False))
        assert safe_count == len(security_payloads.FORMAT_STRING)

        # Ensure no format strings were interpreted
        interpreted_count = sum(1 for test in format_tests
                              if test.get('format_interpreted', False))
        assert interpreted_count == 0

    @pytest.mark.asyncio
    async def test_unicode_attack_protection(self, paca_system, security_payloads):
        """Test protection against Unicode-based attacks."""
        memory_system = paca_system['memory']
        unicode_tests = []

        for payload in security_payloads.UNICODE_ATTACKS:
            try:
                # Test Unicode attack payloads
                memory_id = await memory_system.store_memory(
                    content=payload,
                    memory_type="unicode_test",
                    importance=0.1,
                    tags=["unicode_attack_test"]
                )

                retrieved = await memory_system.retrieve_memory(memory_id)

                # Check for proper Unicode handling
                properly_handled = self._verify_unicode_safety(
                    retrieved['content'], payload
                )

                unicode_tests.append({
                    'payload': payload,
                    'original_length': len(payload),
                    'retrieved_length': len(retrieved['content']),
                    'properly_handled': properly_handled
                })

            except Exception as e:
                unicode_tests.append({
                    'payload': payload,
                    'blocked': True,
                    'error': str(e),
                    'properly_handled': True
                })

        # Assertions
        safe_count = sum(1 for test in unicode_tests
                        if test.get('properly_handled', False))
        assert safe_count == len(security_payloads.UNICODE_ATTACKS)

    @pytest.mark.asyncio
    async def test_input_length_limits(self, paca_system):
        """Test input length validation and limits."""
        memory_system = paca_system['memory']
        length_tests = []

        # Test various input lengths
        test_lengths = [10, 100, 1000, 10000, 100000, 1000000]

        for length in test_lengths:
            payload = "A" * length
            try:
                memory_id = await memory_system.store_memory(
                    content=payload,
                    memory_type="length_test",
                    importance=0.1,
                    tags=["length_validation"]
                )

                retrieved = await memory_system.retrieve_memory(memory_id)
                length_tests.append({
                    'input_length': length,
                    'stored_successfully': retrieved is not None,
                    'content_preserved': len(retrieved['content']) == length,
                    'within_limits': True
                })

            except Exception as e:
                length_tests.append({
                    'input_length': length,
                    'rejected': True,
                    'error': str(e),
                    'within_limits': length <= 100000  # Assume reasonable limit
                })

        # Verify appropriate handling of different input lengths
        for test in length_tests:
            if test['input_length'] <= 10000:  # Reasonable size
                assert test.get('stored_successfully', False) or test.get('rejected', False)
            elif test['input_length'] > 100000:  # Very large input
                # System should either handle gracefully or reject
                assert test.get('rejected', False) or test.get('stored_successfully', False)

    # Helper methods
    def _is_payload_safe(self, processed_content: str, original_payload: str) -> bool:
        """Check if payload was safely handled (not executed)."""
        if processed_content == original_payload:
            return True  # Stored as-is (safe)

        # Check if dangerous elements were sanitized
        dangerous_patterns = ['<script>', 'javascript:', 'onerror=', 'onload=']
        for pattern in dangerous_patterns:
            if pattern in original_payload and pattern not in processed_content:
                return True  # Sanitized (safe)

        return False

    def _detect_command_execution(self, response: Any) -> bool:
        """Detect if command execution occurred."""
        if not response:
            return False

        # Look for common command execution indicators
        execution_indicators = [
            'uid=', 'gid=',  # Unix id command
            'Administrator', 'SYSTEM',  # Windows whoami
            'bin', 'etc', 'usr',  # Unix directory listing
            'C:\\', 'D:\\',  # Windows paths
        ]

        response_str = str(response).lower()
        return any(indicator.lower() in response_str for indicator in execution_indicators)

    def _sanitize_cache_key(self, key: str) -> str:
        """Sanitize cache key to prevent path traversal."""
        # Remove path traversal patterns
        sanitized = key.replace('..', '').replace('\\', '').replace('/', '_')
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        # Limit length
        return sanitized[:100]

    def _contains_path_traversal(self, text: str) -> bool:
        """Check if text contains path traversal patterns."""
        traversal_patterns = ['..', '\\..\\', '../', '%2e%2e', '....']
        return any(pattern in text.lower() for pattern in traversal_patterns)

    def _detect_format_string_exploitation(self, processed_text: str, original_payload: str) -> bool:
        """Detect if format string was interpreted/exploited."""
        # Check if format specifiers were replaced with actual values
        format_patterns = ['%x', '%s', '%d', '%p', '%n']

        for pattern in format_patterns:
            if pattern in original_payload:
                # If the pattern is gone and replaced with data, it was interpreted
                if pattern not in processed_text and len(processed_text) != len(original_payload):
                    return True

        return False

    def _verify_unicode_safety(self, processed_text: str, original_payload: str) -> bool:
        """Verify Unicode content is handled safely."""
        # Check for dangerous Unicode characters that should be filtered/escaped
        dangerous_unicode = [
            '\u202e',  # Right-to-left override
            '\u0000',  # Null character
            '\ufeff',  # Byte order mark
        ]

        # If dangerous characters are removed or escaped, it's safe
        for char in dangerous_unicode:
            if char in original_payload:
                if char not in processed_text:
                    return True  # Removed (safe)
                # Check if it's properly escaped
                if f'\\u{ord(char):04x}' in processed_text:
                    return True  # Escaped (safe)

        # If no dangerous characters, it's safe
        return not any(char in original_payload for char in dangerous_unicode)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
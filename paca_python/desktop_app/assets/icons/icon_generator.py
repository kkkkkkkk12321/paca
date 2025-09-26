"""
Icon Generator for PACA Desktop Application
Purpose: Generate SVG icons dynamically for the PACA desktop interface
Author: PACA Development Team
Created: 2024-09-24
"""

from typing import Dict, Tuple, Optional
import base64
from io import BytesIO
import os


class IconGenerator:
    """
    Dynamic SVG icon generation system for PACA desktop application.

    Generates icons in multiple sizes and formats with customizable colors.
    All icons follow Material Design principles for consistency.
    """

    def __init__(self):
        self.icon_cache = {}
        self.color_schemes = {
            'light': {
                'primary': '#2196F3',
                'secondary': '#FFC107',
                'success': '#4CAF50',
                'warning': '#FF9800',
                'error': '#F44336',
                'text': '#212121',
                'background': '#FAFAFA'
            },
            'dark': {
                'primary': '#64B5F6',
                'secondary': '#FFD54F',
                'success': '#81C784',
                'warning': '#FFB74D',
                'error': '#E57373',
                'text': '#FFFFFF',
                'background': '#121212'
            }
        }

    def generate_app_icon(self, size: int = 64, theme: str = 'light') -> str:
        """
        Generate PACA application main icon.

        Args:
            size: Icon size in pixels
            theme: Color theme ('light' or 'dark')

        Returns:
            SVG string representation of the icon
        """
        colors = self.color_schemes[theme]

        return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:{colors['primary']};stop-opacity:1" />
            <stop offset="100%" style="stop-color:{colors['secondary']};stop-opacity:1" />
        </linearGradient>
    </defs>
    <!-- Background circle -->
    <circle cx="32" cy="32" r="30" fill="url(#grad1)" stroke="{colors['text']}" stroke-width="2"/>
    <!-- PACA letters -->
    <text x="32" y="40" text-anchor="middle" fill="{colors['background']}"
          font-family="Arial, sans-serif" font-size="20" font-weight="bold">P</text>
    <!-- Neural network pattern -->
    <circle cx="18" cy="20" r="2" fill="{colors['background']}" opacity="0.8"/>
    <circle cx="32" cy="18" r="2" fill="{colors['background']}" opacity="0.8"/>
    <circle cx="46" cy="20" r="2" fill="{colors['background']}" opacity="0.8"/>
    <line x1="18" y1="20" x2="32" y2="18" stroke="{colors['background']}" stroke-width="1" opacity="0.6"/>
    <line x1="32" y1="18" x2="46" y2="20" stroke="{colors['background']}" stroke-width="1" opacity="0.6"/>
</svg>"""

    def generate_button_icon(self, icon_type: str, size: int = 24, theme: str = 'light') -> str:
        """
        Generate button icons (start, stop, settings, etc.).

        Args:
            icon_type: Type of button icon ('start', 'stop', 'settings', 'pause', 'refresh')
            size: Icon size in pixels
            theme: Color theme

        Returns:
            SVG string representation
        """
        colors = self.color_schemes[theme]

        icons = {
            'start': f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <polygon points="8,5 19,12 8,19" fill="{colors['success']}" stroke="{colors['text']}" stroke-width="1"/>
            </svg>""",

            'stop': f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="12" height="12" fill="{colors['error']}" stroke="{colors['text']}" stroke-width="1"/>
            </svg>""",

            'pause': f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="4" width="4" height="16" fill="{colors['warning']}" stroke="{colors['text']}" stroke-width="1"/>
                <rect x="14" y="4" width="4" height="16" fill="{colors['warning']}" stroke="{colors['text']}" stroke-width="1"/>
            </svg>""",

            'settings': f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="3" fill="{colors['primary']}" stroke="{colors['text']}" stroke-width="1"/>
                <path d="M12,1L12,1c1.1,0,2,0.9,2,2v2c1.7,0.3,3.3,1,4.6,2l1.4-1.4c0.8-0.8,2-0.8,2.8,0l0,0c0.8,0.8,0.8,2,0,2.8L21.4,10
                         c1,1.3,1.7,2.9,2,4.6h2c1.1,0,2,0.9,2,2v0c0,1.1-0.9,2-2,2h-2c-0.3,1.7-1,3.3-2,4.6l1.4,1.4c0.8,0.8,0.8,2,0,2.8l0,0
                         c-0.8,0.8-2,0.8-2.8,0L18.6,21c-1.3,1-2.9,1.7-4.6,2v2c0,1.1-0.9,2-2,2h0c-1.1,0-2-0.9-2-2v-2c-1.7-0.3-3.3-1-4.6-2l-1.4,1.4
                         c-0.8,0.8-2,0.8-2.8,0l0,0c-0.8-0.8-0.8-2,0-2.8L2.6,19c-1-1.3-1.7-2.9-2-4.6H1c-1.1,0-2-0.9-2-2v0c0-1.1,0.9-2,2-2h2
                         c0.3-1.7,1-3.3,2-4.6L3.6,4.4c-0.8-0.8-0.8-2,0-2.8l0,0c0.8-0.8,2-0.8,2.8,0L7.8,3c1.3-1,2.9-1.7,4.6-2V1C12.6,0.4,12.3,1,12,1z"
                      fill="none" stroke="{colors['primary']}" stroke-width="1.5"/>
            </svg>""",

            'refresh': f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M17.65,6.35C16.2,4.9,14.21,4,12,4c-4.42,0-7.99,3.58-7.99,8s3.57,8,7.99,8c3.73,0,6.84-2.55,7.73-6h-2.08
                         c-0.82,2.33-3.04,4-5.65,4c-3.31,0-6-2.69-6-6s2.69-6,6-6c1.66,0,3.14,0.69,4.22,1.78L13,11h7V4L17.65,6.35z"
                      fill="{colors['primary']}" stroke="{colors['text']}" stroke-width="1"/>
            </svg>"""
        }

        return icons.get(icon_type, icons['settings'])

    def generate_status_icon(self, status: str, size: int = 16, theme: str = 'light') -> str:
        """
        Generate status indicator icons.

        Args:
            status: Status type ('active', 'inactive', 'error', 'warning', 'loading')
            size: Icon size in pixels
            theme: Color theme

        Returns:
            SVG string representation
        """
        colors = self.color_schemes[theme]

        icons = {
            'active': f"""<svg width="{size}" height="{size}" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                <circle cx="8" cy="8" r="6" fill="{colors['success']}" stroke="{colors['text']}" stroke-width="1"/>
                <circle cx="8" cy="8" r="3" fill="{colors['background']}" opacity="0.8"/>
            </svg>""",

            'inactive': f"""<svg width="{size}" height="{size}" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                <circle cx="8" cy="8" r="6" fill="{colors['text']}" opacity="0.3" stroke="{colors['text']}" stroke-width="1"/>
            </svg>""",

            'error': f"""<svg width="{size}" height="{size}" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                <circle cx="8" cy="8" r="6" fill="{colors['error']}" stroke="{colors['text']}" stroke-width="1"/>
                <text x="8" y="12" text-anchor="middle" fill="{colors['background']}" font-size="10" font-weight="bold">!</text>
            </svg>""",

            'warning': f"""<svg width="{size}" height="{size}" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                <polygon points="8,2 14,14 2,14" fill="{colors['warning']}" stroke="{colors['text']}" stroke-width="1"/>
                <text x="8" y="12" text-anchor="middle" fill="{colors['text']}" font-size="8" font-weight="bold">!</text>
            </svg>""",

            'loading': f"""<svg width="{size}" height="{size}" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                <circle cx="8" cy="8" r="6" fill="none" stroke="{colors['primary']}" stroke-width="2" stroke-dasharray="10 5">
                    <animateTransform attributeName="transform" type="rotate" values="0 8 8;360 8 8"
                                    dur="1s" repeatCount="indefinite"/>
                </circle>
            </svg>"""
        }

        return icons.get(status, icons['inactive'])

    def save_icon_to_file(self, icon_svg: str, filename: str, directory: str) -> bool:
        """
        Save SVG icon to file.

        Args:
            icon_svg: SVG string content
            filename: Output filename
            directory: Target directory

        Returns:
            Success status
        """
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(icon_svg)

            return True
        except Exception as e:
            print(f"Error saving icon {filename}: {e}")
            return False

    def generate_all_app_icons(self, base_dir: str) -> Dict[str, bool]:
        """
        Generate all application icons in multiple sizes.

        Args:
            base_dir: Base directory for saving icons

        Returns:
            Dictionary of generation results
        """
        results = {}
        app_dir = os.path.join(base_dir, 'app')

        sizes = [16, 32, 64, 128, 256]
        themes = ['light', 'dark']

        for theme in themes:
            for size in sizes:
                icon_svg = self.generate_app_icon(size, theme)
                filename = f"paca_{size}_{theme}.svg"
                success = self.save_icon_to_file(icon_svg, filename, app_dir)
                results[filename] = success

        return results

    def generate_all_button_icons(self, base_dir: str) -> Dict[str, bool]:
        """
        Generate all button icons.

        Args:
            base_dir: Base directory for saving icons

        Returns:
            Dictionary of generation results
        """
        results = {}
        buttons_dir = os.path.join(base_dir, 'buttons')

        button_types = ['start', 'stop', 'pause', 'settings', 'refresh']
        themes = ['light', 'dark']
        sizes = [16, 24, 32]

        for theme in themes:
            for button_type in button_types:
                for size in sizes:
                    icon_svg = self.generate_button_icon(button_type, size, theme)
                    filename = f"{button_type}_{size}_{theme}.svg"
                    success = self.save_icon_to_file(icon_svg, filename, buttons_dir)
                    results[filename] = success

        return results

    def generate_all_status_icons(self, base_dir: str) -> Dict[str, bool]:
        """
        Generate all status icons.

        Args:
            base_dir: Base directory for saving icons

        Returns:
            Dictionary of generation results
        """
        results = {}
        status_dir = os.path.join(base_dir, 'status')

        status_types = ['active', 'inactive', 'error', 'warning', 'loading']
        themes = ['light', 'dark']
        sizes = [12, 16, 20, 24]

        for theme in themes:
            for status in status_types:
                for size in sizes:
                    icon_svg = self.generate_status_icon(status, size, theme)
                    filename = f"{status}_{size}_{theme}.svg"
                    success = self.save_icon_to_file(icon_svg, filename, status_dir)
                    results[filename] = success

        return results


def main():
    """Main function to generate all icons."""
    generator = IconGenerator()

    # Base icons directory
    icons_base = os.path.dirname(os.path.abspath(__file__))

    print("PACA Icon Generation System Starting...")

    # Generate app icons
    print("Generating app icons...")
    app_results = generator.generate_all_app_icons(icons_base)
    app_success = sum(app_results.values())
    print(f"   App icons generated: {app_success}/{len(app_results)}")

    # Generate button icons
    print("Generating button icons...")
    button_results = generator.generate_all_button_icons(icons_base)
    button_success = sum(button_results.values())
    print(f"   Button icons generated: {button_success}/{len(button_results)}")

    # Generate status icons
    print("Generating status icons...")
    status_results = generator.generate_all_status_icons(icons_base)
    status_success = sum(status_results.values())
    print(f"   Status icons generated: {status_success}/{len(status_results)}")

    total_success = app_success + button_success + status_success
    total_icons = len(app_results) + len(button_results) + len(status_results)

    print(f"\nTotal icons generated: {total_success}/{total_icons}")
    print("PACA Icon System Ready!")


if __name__ == "__main__":
    main()
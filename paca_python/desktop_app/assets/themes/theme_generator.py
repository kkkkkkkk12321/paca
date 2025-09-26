"""
Theme Generator for PACA Desktop Application
Purpose: Generate CSS theme files for light, dark, and custom themes
Author: PACA Development Team
Created: 2024-09-24
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ThemeType(Enum):
    """Available theme types."""
    LIGHT = "light"
    DARK = "dark"
    CUSTOM = "custom"


@dataclass
class ColorPalette:
    """Color palette for a theme."""
    # Primary colors
    primary: str
    primary_dark: str
    primary_light: str
    secondary: str
    secondary_dark: str
    secondary_light: str

    # Background colors
    background: str
    surface: str
    card: str
    header: str
    sidebar: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_hint: str

    # State colors
    success: str
    warning: str
    error: str
    info: str

    # Interactive colors
    hover: str
    active: str
    focus: str
    disabled: str

    # Border colors
    border: str
    border_light: str
    border_dark: str
    divider: str

    # Shadow colors
    shadow: str
    shadow_light: str
    shadow_dark: str


@dataclass
class Typography:
    """Typography settings for a theme."""
    # Font families
    font_primary: str
    font_secondary: str
    font_monospace: str

    # Font sizes
    font_size_xs: str
    font_size_sm: str
    font_size_md: str
    font_size_lg: str
    font_size_xl: str
    font_size_2xl: str

    # Font weights
    font_weight_light: str
    font_weight_normal: str
    font_weight_medium: str
    font_weight_bold: str

    # Line heights
    line_height_xs: str
    line_height_sm: str
    line_height_md: str
    line_height_lg: str


@dataclass
class Spacing:
    """Spacing system for a theme."""
    xs: str
    sm: str
    md: str
    lg: str
    xl: str
    xxl: str


@dataclass
class Borders:
    """Border system for a theme."""
    radius_xs: str
    radius_sm: str
    radius_md: str
    radius_lg: str
    radius_xl: str
    width_thin: str
    width_normal: str
    width_thick: str


@dataclass
class Animation:
    """Animation settings for a theme."""
    duration_fast: str
    duration_normal: str
    duration_slow: str
    easing_standard: str
    easing_decelerate: str
    easing_accelerate: str


class ThemeGenerator:
    """
    Generate CSS theme files for PACA desktop application.

    Supports light, dark, and custom themes with comprehensive design tokens.
    """

    def __init__(self):
        self.themes = {}
        self._initialize_default_themes()

    def _initialize_default_themes(self):
        """Initialize default light and dark themes."""
        # Light theme
        light_colors = ColorPalette(
            primary="#2196F3",
            primary_dark="#1976D2",
            primary_light="#64B5F6",
            secondary="#FFC107",
            secondary_dark="#F57C00",
            secondary_light="#FFD54F",
            background="#FAFAFA",
            surface="#FFFFFF",
            card="#FFFFFF",
            header="#F5F5F5",
            sidebar="#F9F9F9",
            text_primary="#212121",
            text_secondary="#757575",
            text_disabled="#BDBDBD",
            text_hint="#9E9E9E",
            success="#4CAF50",
            warning="#FF9800",
            error="#F44336",
            info="#2196F3",
            hover="#F5F5F5",
            active="#EEEEEE",
            focus="#E3F2FD",
            disabled="#F5F5F5",
            border="#E0E0E0",
            border_light="#F0F0F0",
            border_dark="#BDBDBD",
            divider="#E0E0E0",
            shadow="rgba(0, 0, 0, 0.2)",
            shadow_light="rgba(0, 0, 0, 0.1)",
            shadow_dark="rgba(0, 0, 0, 0.3)"
        )

        # Dark theme
        dark_colors = ColorPalette(
            primary="#64B5F6",
            primary_dark="#42A5F5",
            primary_light="#90CAF9",
            secondary="#FFD54F",
            secondary_dark="#FFCA28",
            secondary_light="#FFE082",
            background="#121212",
            surface="#1E1E1E",
            card="#2D2D2D",
            header="#1F1F1F",
            sidebar="#1A1A1A",
            text_primary="#FFFFFF",
            text_secondary="#BBBBBB",
            text_disabled="#666666",
            text_hint="#888888",
            success="#81C784",
            warning="#FFB74D",
            error="#E57373",
            info="#64B5F6",
            hover="#333333",
            active="#404040",
            focus="#1A237E",
            disabled="#333333",
            border="#404040",
            border_light="#555555",
            border_dark="#2D2D2D",
            divider="#404040",
            shadow="rgba(0, 0, 0, 0.5)",
            shadow_light="rgba(0, 0, 0, 0.3)",
            shadow_dark="rgba(0, 0, 0, 0.7)"
        )

        # Common typography
        typography = Typography(
            font_primary="'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
            font_secondary="'Segoe UI', 'Roboto', sans-serif",
            font_monospace="'Cascadia Code', 'Consolas', 'Monaco', monospace",
            font_size_xs="0.75rem",   # 12px
            font_size_sm="0.875rem",  # 14px
            font_size_md="1rem",      # 16px
            font_size_lg="1.125rem",  # 18px
            font_size_xl="1.25rem",   # 20px
            font_size_2xl="1.5rem",   # 24px
            font_weight_light="300",
            font_weight_normal="400",
            font_weight_medium="500",
            font_weight_bold="600",
            line_height_xs="1.2",
            line_height_sm="1.3",
            line_height_md="1.4",
            line_height_lg="1.6"
        )

        # Common spacing
        spacing = Spacing(
            xs="0.25rem",   # 4px
            sm="0.5rem",    # 8px
            md="1rem",      # 16px
            lg="1.5rem",    # 24px
            xl="2rem",      # 32px
            xxl="3rem"      # 48px
        )

        # Common borders
        borders = Borders(
            radius_xs="0.125rem",  # 2px
            radius_sm="0.25rem",   # 4px
            radius_md="0.5rem",    # 8px
            radius_lg="0.75rem",   # 12px
            radius_xl="1rem",      # 16px
            width_thin="1px",
            width_normal="2px",
            width_thick="3px"
        )

        # Common animations
        animation = Animation(
            duration_fast="150ms",
            duration_normal="300ms",
            duration_slow="500ms",
            easing_standard="cubic-bezier(0.4, 0.0, 0.2, 1)",
            easing_decelerate="cubic-bezier(0.0, 0.0, 0.2, 1)",
            easing_accelerate="cubic-bezier(0.4, 0.0, 1, 1)"
        )

        self.themes[ThemeType.LIGHT] = {
            'colors': light_colors,
            'typography': typography,
            'spacing': spacing,
            'borders': borders,
            'animation': animation
        }

        self.themes[ThemeType.DARK] = {
            'colors': dark_colors,
            'typography': typography,
            'spacing': spacing,
            'borders': borders,
            'animation': animation
        }

    def generate_css_variables(self, theme_data: Dict) -> str:
        """
        Generate CSS custom properties (variables) from theme data.

        Args:
            theme_data: Theme configuration data

        Returns:
            CSS variables string
        """
        css_vars = []

        # Colors
        colors = theme_data['colors']
        for key, value in asdict(colors).items():
            css_name = f"--paca-color-{key.replace('_', '-')}"
            css_vars.append(f"  {css_name}: {value};")

        # Typography
        typography = theme_data['typography']
        for key, value in asdict(typography).items():
            css_name = f"--paca-{key.replace('_', '-')}"
            css_vars.append(f"  {css_name}: {value};")

        # Spacing
        spacing = theme_data['spacing']
        for key, value in asdict(spacing).items():
            css_name = f"--paca-spacing-{key}"
            css_vars.append(f"  {css_name}: {value};")

        # Borders
        borders = theme_data['borders']
        for key, value in asdict(borders).items():
            css_name = f"--paca-{key.replace('_', '-')}"
            css_vars.append(f"  {css_name}: {value};")

        # Animation
        animation = theme_data['animation']
        for key, value in asdict(animation).items():
            css_name = f"--paca-{key.replace('_', '-')}"
            css_vars.append(f"  {css_name}: {value};")

        return "\n".join(css_vars)

    def generate_component_styles(self, theme_data: Dict) -> str:
        """
        Generate component-specific CSS styles.

        Args:
            theme_data: Theme configuration data

        Returns:
            Component CSS styles
        """
        return """
/* Application Layout */
.paca-app {
  background-color: var(--paca-color-background);
  color: var(--paca-color-text-primary);
  font-family: var(--paca-font-primary);
  font-size: var(--paca-font-size-md);
  line-height: var(--paca-line-height-md);
}

.paca-header {
  background-color: var(--paca-color-header);
  border-bottom: var(--paca-width-thin) solid var(--paca-color-border);
  padding: var(--paca-spacing-md);
  box-shadow: 0 2px 4px var(--paca-color-shadow-light);
}

.paca-sidebar {
  background-color: var(--paca-color-sidebar);
  border-right: var(--paca-width-thin) solid var(--paca-color-border);
  padding: var(--paca-spacing-lg);
}

.paca-main-content {
  background-color: var(--paca-color-background);
  padding: var(--paca-spacing-lg);
}

/* Cards and Surfaces */
.paca-card {
  background-color: var(--paca-color-card);
  border: var(--paca-width-thin) solid var(--paca-color-border);
  border-radius: var(--paca-radius-md);
  box-shadow: 0 2px 8px var(--paca-color-shadow-light);
  padding: var(--paca-spacing-lg);
  transition: box-shadow var(--paca-duration-normal) var(--paca-easing-standard);
}

.paca-card:hover {
  box-shadow: 0 4px 12px var(--paca-color-shadow);
}

.paca-surface {
  background-color: var(--paca-color-surface);
  border-radius: var(--paca-radius-sm);
}

/* Buttons */
.paca-button {
  background-color: var(--paca-color-primary);
  color: var(--paca-color-surface);
  border: none;
  border-radius: var(--paca-radius-sm);
  padding: var(--paca-spacing-sm) var(--paca-spacing-md);
  font-family: var(--paca-font-primary);
  font-size: var(--paca-font-size-md);
  font-weight: var(--paca-font-weight-medium);
  cursor: pointer;
  transition: all var(--paca-duration-fast) var(--paca-easing-standard);
}

.paca-button:hover {
  background-color: var(--paca-color-primary-dark);
  transform: translateY(-1px);
  box-shadow: 0 4px 8px var(--paca-color-shadow);
}

.paca-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px var(--paca-color-shadow);
}

.paca-button:focus {
  outline: 2px solid var(--paca-color-focus);
  outline-offset: 2px;
}

.paca-button:disabled {
  background-color: var(--paca-color-disabled);
  color: var(--paca-color-text-disabled);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.paca-button-secondary {
  background-color: var(--paca-color-secondary);
  color: var(--paca-color-text-primary);
}

.paca-button-secondary:hover {
  background-color: var(--paca-color-secondary-dark);
}

.paca-button-outline {
  background-color: transparent;
  color: var(--paca-color-primary);
  border: var(--paca-width-normal) solid var(--paca-color-primary);
}

.paca-button-outline:hover {
  background-color: var(--paca-color-primary);
  color: var(--paca-color-surface);
}

/* Input Fields */
.paca-input {
  background-color: var(--paca-color-surface);
  color: var(--paca-color-text-primary);
  border: var(--paca-width-thin) solid var(--paca-color-border);
  border-radius: var(--paca-radius-sm);
  padding: var(--paca-spacing-sm) var(--paca-spacing-md);
  font-family: var(--paca-font-primary);
  font-size: var(--paca-font-size-md);
  transition: border-color var(--paca-duration-fast) var(--paca-easing-standard);
}

.paca-input:focus {
  outline: none;
  border-color: var(--paca-color-primary);
  box-shadow: 0 0 0 3px var(--paca-color-focus);
}

.paca-input:disabled {
  background-color: var(--paca-color-disabled);
  color: var(--paca-color-text-disabled);
  cursor: not-allowed;
}

.paca-input::placeholder {
  color: var(--paca-color-text-hint);
}

/* Status Indicators */
.paca-status-success {
  color: var(--paca-color-success);
}

.paca-status-warning {
  color: var(--paca-color-warning);
}

.paca-status-error {
  color: var(--paca-color-error);
}

.paca-status-info {
  color: var(--paca-color-info);
}

/* Navigation */
.paca-nav-item {
  display: flex;
  align-items: center;
  padding: var(--paca-spacing-sm) var(--paca-spacing-md);
  margin-bottom: var(--paca-spacing-xs);
  border-radius: var(--paca-radius-sm);
  color: var(--paca-color-text-secondary);
  text-decoration: none;
  transition: all var(--paca-duration-fast) var(--paca-easing-standard);
}

.paca-nav-item:hover {
  background-color: var(--paca-color-hover);
  color: var(--paca-color-text-primary);
}

.paca-nav-item.active {
  background-color: var(--paca-color-primary);
  color: var(--paca-color-surface);
}

/* Scrollbars */
.paca-scrollbar::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.paca-scrollbar::-webkit-scrollbar-track {
  background-color: var(--paca-color-background);
}

.paca-scrollbar::-webkit-scrollbar-thumb {
  background-color: var(--paca-color-border);
  border-radius: var(--paca-radius-sm);
}

.paca-scrollbar::-webkit-scrollbar-thumb:hover {
  background-color: var(--paca-color-border-dark);
}

/* Dividers */
.paca-divider {
  border: none;
  height: var(--paca-width-thin);
  background-color: var(--paca-color-divider);
  margin: var(--paca-spacing-md) 0;
}

/* Loading States */
.paca-loading {
  opacity: 0.6;
  pointer-events: none;
}

.paca-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--paca-color-border);
  border-top: 2px solid var(--paca-color-primary);
  border-radius: 50%;
  animation: paca-spin var(--paca-duration-slow) linear infinite;
}

@keyframes paca-spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
  .paca-card {
    padding: var(--paca-spacing-md);
    margin: var(--paca-spacing-sm);
  }

  .paca-button {
    padding: var(--paca-spacing-md) var(--paca-spacing-lg);
    font-size: var(--paca-font-size-md);
  }
}

@media (max-width: 480px) {
  .paca-header {
    padding: var(--paca-spacing-sm);
  }

  .paca-main-content {
    padding: var(--paca-spacing-md);
  }
}
"""

    def generate_theme_css(self, theme_type: ThemeType) -> str:
        """
        Generate complete CSS theme file.

        Args:
            theme_type: Type of theme to generate

        Returns:
            Complete CSS theme content
        """
        if theme_type not in self.themes:
            raise ValueError(f"Theme type {theme_type.value} not found")

        theme_data = self.themes[theme_type]

        header = f"""/*
 * PACA Desktop Application - {theme_type.value.title()} Theme
 * Generated automatically by ThemeGenerator
 *
 * This file contains all design tokens and component styles
 * for the {theme_type.value} theme of PACA desktop application.
 */

"""

        # Root variables
        root_section = f":root {{\n{self.generate_css_variables(theme_data)}\n}}\n\n"

        # Component styles
        components_section = self.generate_component_styles(theme_data)

        return header + root_section + components_section

    def save_theme_files(self, base_dir: str) -> Dict[str, bool]:
        """
        Save all theme CSS files.

        Args:
            base_dir: Base directory for themes

        Returns:
            Dictionary of save results
        """
        results = {}

        for theme_type in [ThemeType.LIGHT, ThemeType.DARK]:
            try:
                css_content = self.generate_theme_css(theme_type)
                theme_dir = os.path.join(base_dir, theme_type.value)
                os.makedirs(theme_dir, exist_ok=True)

                # Save main theme file
                theme_file = os.path.join(theme_dir, f"{theme_type.value}.css")
                with open(theme_file, 'w', encoding='utf-8') as f:
                    f.write(css_content)
                results[f"{theme_type.value}/{theme_type.value}.css"] = True

                # Save variables only file
                vars_file = os.path.join(theme_dir, f"{theme_type.value}-vars.css")
                vars_content = f":root {{\n{self.generate_css_variables(self.themes[theme_type])}\n}}\n"
                with open(vars_file, 'w', encoding='utf-8') as f:
                    f.write(vars_content)
                results[f"{theme_type.value}/{theme_type.value}-vars.css"] = True

            except Exception as e:
                print(f"Error generating {theme_type.value} theme: {e}")
                results[f"{theme_type.value}/{theme_type.value}.css"] = False
                results[f"{theme_type.value}/{theme_type.value}-vars.css"] = False

        # Generate custom theme template
        try:
            custom_dir = os.path.join(base_dir, "custom")
            os.makedirs(custom_dir, exist_ok=True)

            # Create custom theme template
            template_content = self._generate_custom_theme_template()
            template_file = os.path.join(custom_dir, "custom-template.css")
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
            results["custom/custom-template.css"] = True

            # Create theme configuration JSON
            config_content = self._generate_theme_config()
            config_file = os.path.join(custom_dir, "theme-config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            results["custom/theme-config.json"] = True

        except Exception as e:
            print(f"Error generating custom theme template: {e}")
            results["custom/custom-template.css"] = False
            results["custom/theme-config.json"] = False

        return results

    def _generate_custom_theme_template(self) -> str:
        """Generate custom theme template with placeholders."""
        return """/*
 * PACA Desktop Application - Custom Theme Template
 *
 * Copy this file to create your own custom theme.
 * Replace the color values with your preferred colors.
 *
 * Usage:
 * 1. Copy this file and rename it (e.g., my-theme.css)
 * 2. Modify the color values below
 * 3. Load the theme in your PACA application
 */

:root {
  /* Primary Colors - Main brand colors */
  --paca-color-primary: #YOUR_PRIMARY_COLOR;
  --paca-color-primary-dark: #YOUR_PRIMARY_DARK_COLOR;
  --paca-color-primary-light: #YOUR_PRIMARY_LIGHT_COLOR;
  --paca-color-secondary: #YOUR_SECONDARY_COLOR;
  --paca-color-secondary-dark: #YOUR_SECONDARY_DARK_COLOR;
  --paca-color-secondary-light: #YOUR_SECONDARY_LIGHT_COLOR;

  /* Background Colors - App backgrounds */
  --paca-color-background: #YOUR_BACKGROUND_COLOR;
  --paca-color-surface: #YOUR_SURFACE_COLOR;
  --paca-color-card: #YOUR_CARD_COLOR;
  --paca-color-header: #YOUR_HEADER_COLOR;
  --paca-color-sidebar: #YOUR_SIDEBAR_COLOR;

  /* Text Colors - Text hierarchy */
  --paca-color-text-primary: #YOUR_PRIMARY_TEXT_COLOR;
  --paca-color-text-secondary: #YOUR_SECONDARY_TEXT_COLOR;
  --paca-color-text-disabled: #YOUR_DISABLED_TEXT_COLOR;
  --paca-color-text-hint: #YOUR_HINT_TEXT_COLOR;

  /* State Colors - Status indicators */
  --paca-color-success: #YOUR_SUCCESS_COLOR;
  --paca-color-warning: #YOUR_WARNING_COLOR;
  --paca-color-error: #YOUR_ERROR_COLOR;
  --paca-color-info: #YOUR_INFO_COLOR;

  /* Interactive Colors - UI interactions */
  --paca-color-hover: #YOUR_HOVER_COLOR;
  --paca-color-active: #YOUR_ACTIVE_COLOR;
  --paca-color-focus: #YOUR_FOCUS_COLOR;
  --paca-color-disabled: #YOUR_DISABLED_COLOR;

  /* Border Colors - Lines and dividers */
  --paca-color-border: #YOUR_BORDER_COLOR;
  --paca-color-border-light: #YOUR_LIGHT_BORDER_COLOR;
  --paca-color-border-dark: #YOUR_DARK_BORDER_COLOR;
  --paca-color-divider: #YOUR_DIVIDER_COLOR;

  /* Shadow Colors - Depth and elevation */
  --paca-color-shadow: rgba(0, 0, 0, 0.2);
  --paca-color-shadow-light: rgba(0, 0, 0, 0.1);
  --paca-color-shadow-dark: rgba(0, 0, 0, 0.3);
}

/*
 * Example Custom Theme - "Ocean Breeze"
 * Uncomment and modify these values for a sample ocean-themed palette
 */
/*
:root {
  --paca-color-primary: #0077BE;
  --paca-color-primary-dark: #005A8B;
  --paca-color-primary-light: #4A9BE0;
  --paca-color-secondary: #00A86B;
  --paca-color-secondary-dark: #007F52;
  --paca-color-secondary-light: #4DC995;

  --paca-color-background: #F0F8FF;
  --paca-color-surface: #FFFFFF;
  --paca-color-card: #FAFFFE;
  --paca-color-header: #E8F4F8;
  --paca-color-sidebar: #F0F8FF;

  --paca-color-text-primary: #1B4965;
  --paca-color-text-secondary: #5FA8D3;
  --paca-color-text-disabled: #BEE9E8;
  --paca-color-text-hint: #8ECAE6;

  --paca-color-success: #06FFA5;
  --paca-color-warning: #FFB700;
  --paca-color-error: #FF006E;
  --paca-color-info: #8ECAE6;

  --paca-color-hover: #E8F4F8;
  --paca-color-active: #D1ECF1;
  --paca-color-focus: #B6E5F0;
  --paca-color-disabled: #F0F8FF;

  --paca-color-border: #8ECAE6;
  --paca-color-border-light: #BEE9E8;
  --paca-color-border-dark: #5FA8D3;
  --paca-color-divider: #8ECAE6;

  --paca-color-shadow: rgba(27, 73, 101, 0.2);
  --paca-color-shadow-light: rgba(27, 73, 101, 0.1);
  --paca-color-shadow-dark: rgba(27, 73, 101, 0.3);
}
*/
"""

    def _generate_theme_config(self) -> str:
        """Generate theme configuration JSON."""
        config = {
            "theme_system": {
                "version": "1.0.0",
                "description": "PACA Desktop Application Theme Configuration",
                "supported_themes": ["light", "dark", "custom"]
            },
            "color_categories": {
                "primary": ["primary", "primary-dark", "primary-light"],
                "secondary": ["secondary", "secondary-dark", "secondary-light"],
                "background": ["background", "surface", "card", "header", "sidebar"],
                "text": ["text-primary", "text-secondary", "text-disabled", "text-hint"],
                "state": ["success", "warning", "error", "info"],
                "interactive": ["hover", "active", "focus", "disabled"],
                "border": ["border", "border-light", "border-dark", "divider"],
                "shadow": ["shadow", "shadow-light", "shadow-dark"]
            },
            "custom_theme_instructions": [
                "1. Copy custom-template.css to create your custom theme",
                "2. Replace placeholder values with your color choices",
                "3. Ensure sufficient contrast for accessibility",
                "4. Test your theme with the PACA application",
                "5. Save your theme file in the custom themes folder"
            ],
            "accessibility_guidelines": {
                "contrast_ratio_normal": 4.5,
                "contrast_ratio_large": 3.0,
                "focus_indicators": "Always provide visible focus indicators",
                "color_independence": "Do not rely solely on color to convey information"
            }
        }
        return json.dumps(config, indent=2)


def main():
    """Main function to generate all theme files."""
    generator = ThemeGenerator()

    # Base themes directory
    themes_base = os.path.dirname(os.path.abspath(__file__))

    print("PACA Theme Generation System Starting...")

    try:
        results = generator.save_theme_files(themes_base)

        # Count results by category
        light_success = sum(1 for k, v in results.items() if k.startswith("light/") and v)
        dark_success = sum(1 for k, v in results.items() if k.startswith("dark/") and v)
        custom_success = sum(1 for k, v in results.items() if k.startswith("custom/") and v)

        total_success = sum(results.values())
        total_files = len(results)

        print(f"Light theme files generated: {light_success}/2")
        print(f"Dark theme files generated: {dark_success}/2")
        print(f"Custom theme files generated: {custom_success}/2")
        print(f"\nTotal theme files generated: {total_success}/{total_files}")
        print("PACA Theme System Ready!")

    except Exception as e:
        print(f"Error generating themes: {e}")


if __name__ == "__main__":
    main()
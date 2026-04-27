# Reasoning Viewer Design System

This document specifies the design patterns, color schemes, and component styles used in the Little Steer Reasoning Viewer. These guidelines should be followed when building future viewers or dashboard sites in this repository to ensure a consistent, premium aesthetic.

---

## 1. Color Palette: Everforest (Medium Contrast)

The system uses the **Everforest** color scheme, providing high readability and a "soft" feel. All pages must support both Dark and Light variants using CSS custom properties.

### Everforest Dark
```css
[data-theme="dark"] {
  --bg:        #2d353b; /* Base background */
  --bg-dim:    #232a2e; /* Deep background (body) */
  --bg-1:      #343f44; /* Secondary background (cards/navbar) */
  --bg-2:      #3d484d; /* Tertiary background (hover/active) */
  --bg-3:      #475258; /* Borders / Disabled states */
  --fg:        #d3c6aa; /* Main text */
  --fg-dim:    #9da9a0; /* Secondary/Muted text */
  --red:       #e67e80;
  --orange:    #e69875;
  --yellow:    #dbbc7f;
  --green:     #a7c080;
  --aqua:      #83c092;
  --blue:      #7fbbb3;
  --purple:    #d699b6;
  --grey:      #7a8478;
}
```

### Everforest Light
```css
[data-theme="light"] {
  --bg:        #fdf6e3;
  --bg-dim:    #f4f0d9;
  --bg-1:      #e9e4ca;
  --bg-2:      #ddd8be;
  --bg-3:      #cac9ad;
  --fg:        #5c6a72;
  --fg-dim:    #829181;
  --red:       #f85552;
  --orange:    #f57d26;
  --yellow:    #dfa000;
  --green:     #8da101;
  --aqua:      #35a77c;
  --blue:      #3a94c5;
  --purple:    #df69ba;
  --grey:      #a6b0a0;
}
```

---

## 2. Typography

- **Primary Font**: [Inter](https://fonts.google.com/specimen/Inter) (Google Fonts).
- **Fallback**: `system-ui, sans-serif`.
- **Code Font**: `JetBrains Mono`, `Fira Code`, or `monospace`.
- **Body Size**: `15px` with `1.6` line-height.
- **Weights**: Regular (400), Medium (500), Semi-Bold (600), Bold (700).

---

## 3. Core Layout Components

### Container
- **Max Width**: `1400px`.
- **Padding**: `1.25rem` horizontal.

### Navbar
- **Position**: `sticky; top: 0; z-index: 200`.
- **Background**: `var(--bg-1)`.
- **Border**: `1px solid var(--bg-3)` (bottom).
- **Shadow**: `0 1px 6px rgba(0,0,0,.25)`.
- **Layout**: Flexbox with `align-items: baseline` to align brand name and sub-text correctly.

### The "Card" Pattern
All content sections (info bars, control panels, filters) should use the card style:
```css
.card {
  background:    var(--bg-1);
  border:        1px solid var(--bg-3);
  border-radius: 8px;
  padding:       16px;
  box-shadow:    0 2px 6px rgba(0, 0, 0, 0.25);
}
```

---

## 4. UI Components

### Badges & Pills
Badges are used for metadata (models, datasets, judges).
- **Base Style**: Rounded (`999px`), bold font weight (`500` or `600`), small size (`0.75rem`).
- **Colors**:
  - **Model**: `color: var(--blue); background: var(--bg-3)`.
  - **Dataset**: `color: var(--green); background: var(--bg-3)`.
  - **Judge**: `color: var(--purple); background: var(--bg-3)`.
  - **Work Order**: `color: var(--orange); background: var(--bg-3); border: 1px solid var(--orange)`.

### Interactive Chips
Used for filtering and comparison toggles.
- **Normal State**: `background: var(--bg-2); border: 1px solid var(--bg-3)`.
- **Active/Open State**: `background: var(--blue); color: var(--bg); font-weight: 600`.
- **Hover**: Subtle opacity or brightness change.

### Data Tables
- **Header**: Caps, letter-spacing `0.05em`, background `var(--bg-2)`.
- **Rows**: `cursor: pointer` for navigation, `transition: background 0.1s`.
- **Hover Row**: `background: var(--bg-2)`.
- **Border**: `1px solid var(--bg-2)` between rows.

---

## 5. Conversation Styling

When displaying LLM outputs or chat traces:

### Turn Headers
- **Roles**: Distinct colors for labels.
  - **User**: `var(--blue)` on `var(--bg-3)`.
  - **Assistant**: `var(--green)` on `var(--bg-3)`.
  - **Thinking**: `var(--fg-dim)` on `var(--bg-2)`.

### Turn Bodies
- **Base**: `padding: 16px`, `border-radius: 8px`, `border: 1px solid var(--bg-3)`.
- **Side Border**: A `3px` solid border on the left helps identify the role visually.
  - User: `border-left-color: var(--blue)`.
  - Assistant: `border-left-color: var(--green)`.
- **Thinking Blocks**: Use `var(--bg)` (base background) to "recess" thinking blocks into the page, making them look internal/background compared to messages.

---

## 6. Interactivity & Transitions

### Theme Switcher
Every page **must** include the standard theme toggle:
1. Store preference in `localStorage.getItem("theme")`.
2. Default to `"dark"`.
3. Apply via `document.documentElement.dataset.theme`.
4. Dispatch a `themechange` event for any charts (e.g., Chart.js) to re-render with appropriate colors.

---

## 7. Charts & Visualizations

The viewer uses [Chart.js](https://www.chartjs.org/) for data visualization, with custom styling to match the Everforest theme.

### Theme-Aware Colors
Charts must react to theme changes using a `chartColors()` helper:
```javascript
function chartColors() {
  const dark = document.documentElement.dataset.theme === "dark";
  return {
    grid:  dark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
    tick:  dark ? "#d3c6aa" : "#5c6a72",
    label: dark ? "#d3c6aa" : "#5c6a72",
  };
}
```

### Categorical Palettes
- **Standard Palette**: `["#dbbc7f", "#e69875", "#e67e80", "#a7c080", "#83c092", "#d699b6", "#7fbbb3", "#d3c6aa"]`
- **Guard Scores**:
  - Harmful/Unsafe: `var(--red)` (`#e67e80`)
  - Safe/Refusal: `var(--green)` (`#a7c080`)
  - Controversial: `var(--yellow)` (`#dbbc7f`)

### Layout Containers
- **Chart Box**: A card containing a header and a canvas. Header background matches the categorical color of the data group.
- **Wide Chart**: Used for horizontal bar charts with many items; height is dynamically calculated (`nItems * 32px + 48px`).
- **Heatmaps**: Implemented via HTML tables with `background: rgba(var(--aqua-rgb), alpha)` where alpha corresponds to value density.

---

## 8. Responsive Design

- **Mobile (< 800px)**:
  - Stack filter bars vertically.
  - Set search inputs to `width: 100%`.
  - Hide non-essential columns in tables (e.g., prompt snippets).
  - Force comparison grids to single columns.

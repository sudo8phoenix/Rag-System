---
name: frontend-design
description: "Create distinctive, production-grade frontend interfaces with high design quality. Use this for web components, pages, landing sites, dashboards, posters, interactive artifacts, React/Vue/HTML-CSS layouts, and styling or beautifying existing UI. Produces bold, cohesive, memorable frontend code with strong typography, intentional color systems, expressive motion, and non-generic visual direction."
argument-hint: "Describe the interface, audience, tech stack, constraints, and desired aesthetic direction."
license: Complete terms in LICENSE.txt
user-invocable: true
disable-model-invocation: false
---

# Frontend Design

Build visually distinctive frontend interfaces that feel intentionally designed, production-ready, and context-specific.

## When To Use
- User asks to build or redesign any UI surface: component, page, application, or full flow.
- User requests better styling, visual polish, branding, or “make it look professional.”
- User needs HTML/CSS/JS, React, Vue, or framework UI implementation with strong creative direction.
- User wants design quality beyond generic templates or repetitive AI-looking outputs.

## Inputs To Collect
Gather these before implementation:
- Product goal: what the interface does and what action it should drive.
- Audience: who will use it and their context.
- Platform and stack: HTML, React, Vue, Next.js, etc.
- Constraints: accessibility, performance, browser support, design system requirements.
- Deliverable scope: single component, page, multi-page flow, or complete app surface.
- Aesthetic intent: one strong direction (minimal, editorial, brutalist, playful, luxury, industrial, retro-futuristic, etc.).

If key inputs are missing, ask concise clarification questions before coding.

## Default Operational Policy
- Stack default: infer from workspace and project context when the user does not specify a framework.
- Asset default: web-hosted fonts/assets are allowed unless the user requests local-only constraints.
- Invocation default: keep this skill auto-load eligible and slash-invocable.

## Decision Logic
1. Existing design system check:
- If the project already has a clear system, preserve and extend that language.
- If no system exists, define a fresh visual language with explicit typography, color, spacing, and motion tokens.

2. Scope and fidelity decision:
- Small scope (single component): prioritize high-impact details and reuse-friendly API.
- Medium scope (single page): define full layout rhythm, hierarchy, and responsive behavior.
- Large scope (application): establish reusable primitives, semantic tokens, and consistent interaction patterns.

3. Aesthetic intensity selection:
- Minimal/refined: fewer elements, tighter spacing discipline, subtle motion.
- Maximal/expressive: layered backgrounds, richer animation choreography, stronger visual contrast.

4. Motion strategy:
- Static utility UIs: subtle feedback-focused motion.
- Brand or marketing UIs: staged reveals, scroll choreography, and hero moments.

## Implementation Workflow
1. Frame the concept
- Write a short concept statement: purpose, tone, and memorable signature element.
- Commit to one dominant design direction and avoid mixed, conflicting styles.

2. Define visual system
- Create CSS variables or theme tokens for colors, spacing, radii, shadows, and animation timing.
- Select distinctive type pairings (display + body) suitable for the chosen tone.
- Ensure contrast and readability are acceptable from the start.

3. Design composition
- Build clear hierarchy with intentional scale changes and whitespace control.
- Introduce asymmetry, overlap, diagonal flow, or structured density where it supports the concept.
- Design backgrounds with atmosphere (gradients, textures, geometric layers, subtle noise) instead of flat defaults.

4. Implement production-grade code
- Deliver real working code, not pseudo-layout snippets.
- Keep structure semantic and maintainable.
- Use reusable classes/components and avoid hardcoded one-off values when tokens are appropriate.
- Match implementation complexity to the concept.

5. Add meaningful motion
- Prioritize one or two signature animations over many minor effects.
- Use staggered entrances, hover/focus transitions, and state changes that reinforce hierarchy.
- Respect reduced-motion preferences for accessibility.

6. Responsive and accessibility pass
- Validate desktop and mobile behavior.
- Check keyboard navigation, focus visibility, semantic structure, and color contrast.
- Ensure interaction targets are usable on touch devices.

7. Quality polish pass
- Remove generic defaults and visual noise.
- Tighten spacing, alignment, and typographic rhythm.
- Ensure the final design feels intentional, cohesive, and memorable.

## Quality Criteria (Definition Of Done)
- The interface is functional and production-grade.
- The design has a clear, recognizable point of view.
- Typography is intentional and non-generic.
- Color and theming are coherent with strong contrast logic.
- Motion supports comprehension and delight without clutter.
- The layout is responsive across common breakpoints.
- Accessibility fundamentals are addressed.
- Output avoids repetitive, cookie-cutter AI aesthetics.

## Output Requirements
When delivering results:
- Provide complete runnable code for the requested stack.
- Include concise implementation notes only when needed for setup or tradeoffs.
- Explain the chosen design direction in 2-4 lines.
- If assumptions were made, list them explicitly.

## Recovery Path
If the first draft feels generic or inconsistent:
- Re-state the design concept in one sentence.
- Replace typography pair and palette with a stronger identity.
- Remove low-impact effects and add one memorable signature interaction.
- Rework spacing/scale rhythm before adding more visual elements.

## Example Prompts
- Build a fintech landing page in React with an editorial luxury tone and strong conversion focus.
- Design a dashboard settings page in plain HTML/CSS with industrial utilitarian styling and excellent accessibility.
- Restyle this existing component to feel playful and tactile without changing its API.
- Create a portfolio hero section that is minimalist but unmistakably original, with one signature motion sequence.

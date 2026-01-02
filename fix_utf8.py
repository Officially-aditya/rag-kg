#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# Read the file
with open('doc.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Dictionary of UTF-8 characters to LaTeX replacements
replacements = {
    # Arrows
    '→': r'$\rightarrow$',
    '←': r'$\leftarrow$',
    '↓': r'$\downarrow$',
    '↑': r'$\uparrow$',

    # Stars and bullets
    '★': r'$\star$',
    '•': r'$\bullet$',

    # Dashes
    '–': '--',
    '—': '---',

    # Math operators
    '≈': r'$\approx$',
    '≤': r'$\leq$',
    '≥': r'$\geq$',
    '≠': r'$\neq$',
    '×': r'$\times$',
    '·': r'$\cdot$',
    '÷': r'$\div$',
    '±': r'$\pm$',

    # Set theory
    '∈': r'$\in$',
    '∉': r'$\notin$',
    '⊂': r'$\subset$',
    '⊆': r'$\subseteq$',
    '∪': r'$\cup$',
    '∩': r'$\cap$',
    '∅': r'$\emptyset$',

    # Calculus/Math
    '∂': r'$\partial$',
    '∫': r'$\int$',
    '∑': r'$\sum$',
    '∏': r'$\prod$',
    '√': r'$\sqrt{}$',
    '∞': r'$\infty$',

    # Logic
    '∧': r'$\land$',
    '∨': r'$\lor$',
    '¬': r'$\neg$',
    '⇒': r'$\Rightarrow$',
    '⇔': r'$\Leftrightarrow$',
    '∀': r'$\forall$',
    '∃': r'$\exists$',

    # Greek letters (lowercase)
    'α': r'$\alpha$',
    'β': r'$\beta$',
    'γ': r'$\gamma$',
    'δ': r'$\delta$',
    'ε': r'$\varepsilon$',
    'θ': r'$\theta$',
    'λ': r'$\lambda$',
    'μ': r'$\mu$',
    'π': r'$\pi$',
    'σ': r'$\sigma$',
    'τ': r'$\tau$',
    'φ': r'$\phi$',
    'ω': r'$\omega$',

    # Greek letters (uppercase)
    'Σ': r'$\Sigma$',
    'Π': r'$\Pi$',
    'Δ': r'$\Delta$',
    'Ω': r'$\Omega$',

    # Superscripts
    '⁰': r'$^0$',
    '¹': r'$^1$',
    '²': r'$^2$',
    '³': r'$^3$',
    '⁴': r'$^4$',
    '⁵': r'$^5$',
    '⁶': r'$^6$',
    '⁷': r'$^7$',
    '⁸': r'$^8$',
    '⁹': r'$^9$',
    '⁺': r'$^+$',
    '⁻': r'$^-$',
    'ⁿ': r'$^n$',
    'ⁱ': r'$^i$',

    # Subscripts
    '₀': r'$_0$',
    '₁': r'$_1$',
    '₂': r'$_2$',
    '₃': r'$_3$',
    '₄': r'$_4$',
    '₅': r'$_5$',
    '₆': r'$_6$',
    '₇': r'$_7$',
    '₈': r'$_8$',
    '₉': r'$_9$',
    '₊': r'$_+$',
    '₋': r'$_-$',
    'ₙ': r'$_n$',
    'ᵢ': r'$_i$',
    'ⱼ': r'$_j$',
    'ₖ': r'$_k$',

    # Special mathematical symbols
    '℃': r'$^\circ$C',
    '℉': r'$^\circ$F',
    '℮': r'e',
    'ℝ': r'$\mathbb{R}$',
    'ℕ': r'$\mathbb{N}$',
    'ℤ': r'$\mathbb{Z}$',
    'ℚ': r'$\mathbb{Q}$',
    'ℂ': r'$\mathbb{C}$',

    # Other operators
    '⊙': r'$\odot$',
    '⊗': r'$\otimes$',
    '⊕': r'$\oplus$',
}

# Apply replacements
for utf8_char, latex_replacement in replacements.items():
    content = content.replace(utf8_char, latex_replacement)

# Write back to file
with open('doc.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("UTF-8 characters replaced with LaTeX equivalents successfully!")

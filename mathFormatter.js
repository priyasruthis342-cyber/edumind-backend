// ─────────────────────────────────────────────────────────────
// mathFormatter.js — Pure Unicode Math & Scientific Notation Cleaner
// Replaces raw LaTeX symbols, $ dollar wrappers, and backslash artifacts
// with clean, readable standard Unicode math notation.
// ─────────────────────────────────────────────────────────────

function replaceFrac(str) {
  let idx = 0;
  while ((idx = str.indexOf('\\frac{', idx)) !== -1) {
    let numStart = idx + 6;
    let depth = 1;
    let i = numStart;
    while (i < str.length && depth > 0) {
      if (str[i] === '{') depth++;
      else if (str[i] === '}') depth--;
      i++;
    }
    if (depth !== 0) { idx += 6; continue; }
    let num = str.slice(numStart, i - 1);
    if (str[i] !== '{') { idx = i; continue; }
    let denStart = i + 1;
    depth = 1;
    i = denStart;
    while (i < str.length && depth > 0) {
      if (str[i] === '{') depth++;
      else if (str[i] === '}') depth--;
      i++;
    }
    if (depth !== 0) { idx = denStart; continue; }
    let den = str.slice(denStart, i - 1);
    let replacement = `(${num}) / (${den})`;
    str = str.slice(0, idx) + replacement + str.slice(i);
    idx += replacement.length;
  }
  return str;
}

function cleanMathFormatting(text) {
  if (!text || typeof text !== 'string') return text;
  let out = text;

  // 1. Convert block math $$...$$
  out = out.replace(/\$\$([\s\S]*?)\$\$/g, (match, inner) => {
    return '\n' + inner.trim() + '\n';
  });

  // 2. Unescape common LaTeX text wrappers
  out = out.replace(/\\(text|mathrm|mathbf|mathit|textbf|mathsf|mathtt)\{([^}]+)\}/g, '$2');

  // 3. Convert \frac{...}{...}
  out = replaceFrac(out);

  // 4. Convert \sqrt
  out = out.replace(/\\sqrt\[([^\]]+)\]\{([^}]+)\}/g, '$1√($2)');
  out = out.replace(/\\sqrt\{([^}]+)\}/g, '√($1)');

  // 5. Escaped braces
  out = out.replace(/\\\{/g, '{').replace(/\\\}/g, '}');

  // 6. Greek & Math symbols
  out = out.replace(/\\infty\b/g, '∞')
           .replace(/\\times\b/g, ' × ')
           .replace(/\\cdot\b/g, ' · ')
           .replace(/\\div\b/g, ' ÷ ')
           .replace(/\\pm\b/g, '±')
           .replace(/\\mp\b/g, '∓')
           .replace(/\\leq\b|\\le\b/g, '≤')
           .replace(/\\geq\b|\\ge\b/g, '≥')
           .replace(/\\neq\b|\\ne\b/g, '≠')
           .replace(/\\approx\b/g, '≈')
           .replace(/\\equiv\b/g, '≡')
           .replace(/\\in\b/g, '∈')
           .replace(/\\notin\b/g, '∉')
           .replace(/\\subset\b/g, '⊂')
           .replace(/\\subseteq\b/g, '⊆')
           .replace(/\\cup\b/g, '∪')
           .replace(/\\cap\b/g, '∩')
           .replace(/\\forall\b/g, '∀')
           .replace(/\\exists\b/g, '∃')
           .replace(/\\sum\b/g, '∑')
           .replace(/\\prod\b/g, '∏')
           .replace(/\\int\b/g, '∫')
           .replace(/\\partial\b/g, '∂')
           .replace(/\\nabla\b/g, '∇')
           .replace(/\\to\b|\\rightarrow\b/g, '→')
           .replace(/\\leftarrow\b/g, '←')
           .replace(/\\implies\b/g, '⇒')
           .replace(/\\iff\b/g, '⇔')
           .replace(/\\log\b/g, 'log')
           .replace(/\\ln\b/g, 'ln')
           .replace(/\\sin\b/g, 'sin')
           .replace(/\\cos\b/g, 'cos')
           .replace(/\\tan\b/g, 'tan')
           .replace(/\\lim\b/g, 'lim')
           .replace(/\\max\b/g, 'max')
           .replace(/\\min\b/g, 'min')
           .replace(/\\theta\b/g, 'θ')
           .replace(/\\Theta\b/g, 'Θ')
           .replace(/\\Omega\b/g, 'Ω')
           .replace(/\\omega\b/g, 'ω')
           .replace(/\\alpha\b/g, 'α')
           .replace(/\\beta\b/g, 'β')
           .replace(/\\gamma\b/g, 'γ')
           .replace(/\\Gamma\b/g, 'Γ')
           .replace(/\\delta\b/g, 'δ')
           .replace(/\\Delta\b/g, 'Δ')
           .replace(/\\epsilon\b/g, 'ε')
           .replace(/\\lambda\b/g, 'λ')
           .replace(/\\Lambda\b/g, 'Λ')
           .replace(/\\mu\b/g, 'μ')
           .replace(/\\pi\b/g, 'π')
           .replace(/\\Pi\b/g, 'Π')
           .replace(/\\rho\b/g, 'ρ')
           .replace(/\\sigma\b/g, 'σ')
           .replace(/\\Sigma\b/g, 'Σ')
           .replace(/\\tau\b/g, 'τ')
           .replace(/\\phi\b/g, 'ϕ')
           .replace(/\\Phi\b/g, 'Φ');

  // 7. Superscripts
  out = out.replace(/\^2\b|\^\{2\}/g, '²')
           .replace(/\^3\b|\^\{3\}/g, '³')
           .replace(/\^0\b|\^\{0\}/g, '⁰')
           .replace(/\^1\b|\^\{1\}/g, '¹')
           .replace(/\^4\b|\^\{4\}/g, '⁴')
           .replace(/\^n\b|\^\{n\}/g, 'ⁿ')
           .replace(/\^k\b|\^\{k\}/g, 'ᵏ')
           .replace(/\^x\b|\^\{x\}/g, 'ˣ')
           .replace(/\^y\b|\^\{y\}/g, 'ʸ')
           .replace(/\^i\b|\^\{i\}/g, 'ⁱ');

  // 8. Subscripts
  out = out.replace(/_0\b|_\{0\}/g, '₀')
           .replace(/_1\b|_\{1\}/g, '₁')
           .replace(/_2\b|_\{2\}/g, '₂')
           .replace(/_3\b|_\{3\}/g, '₃')
           .replace(/_4\b|_\{4\}/g, '₄')
           .replace(/_i\b|_\{i\}/g, 'ᵢ')
           .replace(/_j\b|_\{j\}/g, 'ⱼ')
           .replace(/_k\b|_\{k\}/g, 'ₖ')
           .replace(/_n\b|_\{n\}/g, 'ₙ')
           .replace(/_m\b|_\{m\}/g, 'ₘ');

  // 9. Remove inline math dollar wrappers $...$
  out = out.replace(/\$([^\$\n]+?)\$/g, '$1');

  // 10. Remove LaTeX sizing commands \left \right \big
  out = out.replace(/\\(left|right|big|Big|bigg|Bigg)\b/g, '');

  // 11. Remove any stray $ before letters/numbers (e.g. $u, $v, $x)
  out = out.replace(/\$([a-zA-Z0-9_\+\-\*\/\^\(\)]+)/g, '$1');

  // 12. Strip remaining lone $ symbols
  out = out.replace(/(^|[^\\0-9])\$([^0-9]|$)/g, '$1$2');

  // 13. Clean whitespace
  out = out.replace(/[ \t]{2,}/g, ' ');

  return out.trim();
}

module.exports = {
  cleanMathFormatting
};

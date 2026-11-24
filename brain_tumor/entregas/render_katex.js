// Uso: npm install katex@0.16.11 && node render_katex.js entrega_III.html salida.html  (luego se imprime a PDF)
// Renderiza $$...$$ (bloque) y \(...\) (en línea) con KaTeX y añade su CSS
const fs = require('fs'), path = require('path'), katex = require('katex');
const [src, out] = process.argv.slice(2);
let html = fs.readFileSync(src, 'utf8');
const opts = { throwOnError: true, strict: false };
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => katex.renderToString(tex, { ...opts, displayMode: true }));
html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => katex.renderToString(tex, { ...opts, displayMode: false }));
const css = 'file://' + path.join(__dirname, 'node_modules/katex/dist/katex.min.css');
html = html.replace('</head>', `<link rel="stylesheet" href="${css}">\n</head>`);
fs.writeFileSync(out, html);
console.log('ok');

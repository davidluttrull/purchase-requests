/*
 * Receipts for the purchase request pages: turns photos into clean scanned
 * pages and puts them, with any PDFs, into one PDF.
 *
 *   Receipt.scan(file)      opens the corner editor for a photo. Resolves to
 *                           {blob, url} (a JPEG page) or null if cancelled.
 *   Receipt.pdfInfo(file)   {bytes, pages} for a PDF someone attached.
 *   Receipt.buildPdf(items) one PDF from [{jpeg: Blob} | {pdf: Uint8Array}],
 *                           as base64 for the script.
 *   Receipt.warm()          starts loading the edge finder early.
 *
 * Finding the receipt's edges uses OpenCV (about 10 MB, downloaded the first
 * time a photo is added and cached by the browser after that). Everything
 * else runs without it, so if it can't load, the corners start at the edges
 * of the photo and the person drags them in.
 */
const Receipt = (() => {
  const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js';
  const PDFLIB_URL = 'https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js';
  const MAX_SOURCE = 2600;   // longest side of a photo kept for straightening
  const MAX_OUT = 2200;      // longest side of a finished page
  const MAX_OUT_PIXELS = 3.2e6;
  const DETECT_SIZE = 640;   // longest side used to find the edges
  const PAGE_WIDTH = 612;    // PDF points: 8.5 inches

  // ---- Loading libraries ------------------------------------------------------

  const scripts = {};
  function loadScript(url) {
    if (!scripts[url]) {
      scripts[url] = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = url;
        s.async = true;
        s.onload = resolve;
        s.onerror = () => { delete scripts[url]; reject(new Error('Couldn\u2019t load ' + url)); };
        document.head.appendChild(s);
      });
    }
    return scripts[url];
  }

  let cvReady;
  // Builds of OpenCV.js hand back the library in different ways: ready at
  // once, with a then() that calls back when it has started, or with an
  // onRuntimeInitialized callback. The library object itself can have a
  // then(), so it must never be handed to a promise directly: a promise would
  // keep unwrapping it forever. It travels inside {cv} instead.
  function opencv() {
    if (!cvReady) {
      cvReady = loadScript(OPENCV_URL).then(() => new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('OpenCV didn\u2019t start')), 60000);
        const done = (m) => { clearTimeout(timer); resolve({ cv: m && m.Mat ? m : window.cv }); };
        const cv = window.cv;
        if (!cv) return reject(new Error('OpenCV didn\u2019t load'));
        if (cv.Mat) return done(cv);
        if (typeof cv.then === 'function') return void cv.then(done);
        cv.onRuntimeInitialized = () => done(cv);
      }));
      cvReady.catch(() => { cvReady = null; });
    }
    return cvReady;
  }

  async function pdfLib() {
    await loadScript(PDFLIB_URL);
    return window.PDFLib;
  }

  // ---- Images -------------------------------------------------------------------

  function canvas(w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    return c;
  }

  /** A photo file -> canvas no bigger than MAX_SOURCE, upright (browsers apply the camera's rotation). */
  async function loadImage(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = await new Promise((resolve, reject) => {
        const i = new Image();
        i.onload = () => resolve(i);
        i.onerror = () => reject(new Error('That picture couldn\u2019t be opened. Try a JPEG or PNG, or take the photo again.'));
        i.src = url;
      });
      const scale = Math.min(1, MAX_SOURCE / Math.max(img.naturalWidth, img.naturalHeight));
      const c = canvas(img.naturalWidth * scale, img.naturalHeight * scale);
      const ctx = c.getContext('2d');
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, c.width, c.height);
      return c;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function scaled(src, longest) {
    const s = Math.min(1, longest / Math.max(src.width, src.height));
    const c = canvas(src.width * s, src.height * s);
    const ctx = c.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, c.width, c.height);
    return c;
  }

  // ---- Finding the edges --------------------------------------------------------

  /** [tl, tr, br, bl] as {x, y} in the canvas's pixels, or null if no receipt-shaped outline was found. */
  async function findCorners(src) {
    const cv = (await opencv()).cv;
    const small = scaled(src, DETECT_SIZE);
    const k = src.width / small.width;
    const mats = [];
    const mat = (m) => { mats.push(m); return m; };
    try {
      const img = mat(cv.imread(small));
      const gray = mat(new cv.Mat());
      cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
      const area = small.width * small.height;

      // A light receipt on a darker surface: the brightest large shape.
      const bright = mat(new cv.Mat());
      cv.threshold(gray, bright, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      const kernel = mat(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(9, 9)));
      cv.morphologyEx(bright, bright, cv.MORPH_CLOSE, kernel);
      let quad = largestQuad(cv, bright, area, mat);

      // Otherwise the strongest outline.
      if (!quad) {
        const edges = mat(new cv.Mat());
        cv.Canny(gray, edges, 40, 120);
        const k5 = mat(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5)));
        cv.dilate(edges, edges, k5);
        quad = largestQuad(cv, edges, area, mat);
      }
      return quad ? order(quad.map((p) => ({ x: p.x * k, y: p.y * k }))) : null;
    } finally {
      mats.forEach((m) => { try { m.delete(); } catch (e) { /* already freed */ } });
    }
  }

  function largestQuad(cv, binary, area, mat) {
    const contours = mat(new cv.MatVector());
    const hierarchy = mat(new cv.Mat());
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = null;
    let bestArea = 0;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const a = cv.contourArea(c);
      if (a > bestArea && a > area * 0.1 && a < area * 0.985) {
        const hull = new cv.Mat();
        cv.convexHull(c, hull);
        const approx = new cv.Mat();
        cv.approxPolyDP(hull, approx, 0.02 * cv.arcLength(hull, true), true);
        let pts;
        if (approx.rows === 4) {
          pts = [];
          for (let j = 0; j < 4; j++) pts.push({ x: approx.data32S[j * 2], y: approx.data32S[j * 2 + 1] });
        } else {
          pts = rectPoints(cv.minAreaRect(hull));
        }
        hull.delete();
        approx.delete();
        best = pts;
        bestArea = a;
      }
      c.delete();
    }
    return best;
  }

  function rectPoints(r) {
    const a = r.angle * Math.PI / 180;
    const cos = Math.cos(a) / 2;
    const sin = Math.sin(a) / 2;
    const w = r.size.width;
    const h = r.size.height;
    const cx = r.center.x;
    const cy = r.center.y;
    return [
      { x: cx - w * cos + h * sin, y: cy - w * sin - h * cos },
      { x: cx + w * cos + h * sin, y: cy + w * sin - h * cos },
      { x: cx + w * cos - h * sin, y: cy + w * sin + h * cos },
      { x: cx - w * cos - h * sin, y: cy - w * sin + h * cos },
    ];
  }

  /** Any four points -> [top left, top right, bottom right, bottom left]. */
  function order(pts) {
    const by = (f) => pts.slice().sort((a, b) => f(a) - f(b));
    const tl = by((p) => p.x + p.y)[0];
    const br = by((p) => -(p.x + p.y))[0];
    const tr = by((p) => p.y - p.x)[0];
    const bl = by((p) => p.x - p.y)[0];
    return [tl, tr, br, bl];
  }

  function inset(src, f) {
    const dx = src.width * f;
    const dy = src.height * f;
    return [{ x: dx, y: dy }, { x: src.width - dx, y: dy }, { x: src.width - dx, y: src.height - dy }, { x: dx, y: src.height - dy }];
  }

  // ---- Straightening ------------------------------------------------------------

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  /** The four corners -> a flat, upright rectangle (bilinear sampling through a homography). */
  function warp(src, q) {
    let w = Math.max(dist(q[0], q[1]), dist(q[3], q[2]));
    let h = Math.max(dist(q[0], q[3]), dist(q[1], q[2]));
    const s = Math.min(1, MAX_OUT / Math.max(w, h), Math.sqrt(MAX_OUT_PIXELS / (w * h)));
    w = Math.max(1, Math.round(w * s));
    h = Math.max(1, Math.round(h * s));

    // Unit square -> quad (Heckbert). u along the top edge, v down the left edge.
    const [x0, y0, x1, y1, x2, y2, x3, y3] = [q[0].x, q[0].y, q[1].x, q[1].y, q[2].x, q[2].y, q[3].x, q[3].y];
    const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    const den = dx1 * dy2 - dx2 * dy1 || 1e-9;
    const g = (dx3 * dy2 - dx2 * dy3) / den;
    const hh = (dx1 * dy3 - dx3 * dy1) / den;
    const a = x1 - x0 + g * x1, b = x3 - x0 + hh * x3, c = x0;
    const d = y1 - y0 + g * y1, e = y3 - y0 + hh * y3, f = y0;

    const sw = src.width;
    const sh = src.height;
    const sd = src.getContext('2d').getImageData(0, 0, sw, sh).data;
    const out = canvas(w, h);
    const octx = out.getContext('2d');
    const od = octx.createImageData(w, h);
    const o = od.data;
    for (let j = 0; j < h; j++) {
      const v = (j + 0.5) / h;
      for (let i = 0; i < w; i++) {
        const u = (i + 0.5) / w;
        const z = g * u + hh * v + 1;
        let sx = (a * u + b * v + c) / z - 0.5;
        let sy = (d * u + e * v + f) / z - 0.5;
        sx = sx < 0 ? 0 : sx > sw - 1.001 ? sw - 1.001 : sx;
        sy = sy < 0 ? 0 : sy > sh - 1.001 ? sh - 1.001 : sy;
        const ix = sx | 0, iy = sy | 0, fx = sx - ix, fy = sy - iy;
        const p = (iy * sw + ix) * 4;
        const p2 = p + sw * 4;
        const k = (j * w + i) * 4;
        for (let ch = 0; ch < 3; ch++) {
          const top = sd[p + ch] + (sd[p + 4 + ch] - sd[p + ch]) * fx;
          const bot = sd[p2 + ch] + (sd[p2 + 4 + ch] - sd[p2 + ch]) * fx;
          o[k + ch] = top + (bot - top) * fy;
        }
        o[k + 3] = 255;
      }
    }
    octx.putImageData(od, 0, 0);
    return out;
  }

  function rotate(src, turns) {
    turns = ((turns % 4) + 4) % 4;
    if (!turns) return src;
    const out = turns % 2 ? canvas(src.height, src.width) : canvas(src.width, src.height);
    const ctx = out.getContext('2d');
    ctx.translate(out.width / 2, out.height / 2);
    ctx.rotate(turns * Math.PI / 2);
    ctx.drawImage(src, -src.width / 2, -src.height / 2);
    return out;
  }

  /**
   * The scanned look: evens out shadows by dividing each pixel by how bright
   * the paper is around it, so the paper turns white and the print stays
   * dark. The paper's brightness is the brightest pixel in each block, smoothed
   * across blocks. A block that's all ink (a logo, a barcode) can't be
   * brighter than 60% of the page's usual paper, so it stays dark.
   */
  function scanLook(src) {
    const w = src.width;
    const h = src.height;
    const ctx = src.getContext('2d');
    const img = ctx.getImageData(0, 0, w, h);
    const p = img.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, k = 0; i < gray.length; i++, k += 4) gray[i] = 0.299 * p[k] + 0.587 * p[k + 1] + 0.114 * p[k + 2];

    // Usual paper brightness: the 90th percentile.
    const hist = new Uint32Array(256);
    for (let i = 0; i < gray.length; i += 7) hist[gray[i] | 0]++;
    let seen = 0;
    let paper = 255;
    const want = (gray.length / 7) * 0.9;
    for (let v = 0; v < 256; v++) { seen += hist[v]; if (seen >= want) { paper = v; break; } }
    const floor = Math.max(40, paper * 0.6);

    // Brightest pixel per block, then each block averaged with its neighbours.
    const B = Math.max(16, Math.round(Math.min(w, h) / 14));
    const bw = Math.ceil(w / B);
    const bh = Math.ceil(h / B);
    const peak = new Float32Array(bw * bh);
    for (let y = 0; y < h; y++) {
      const row = ((y / B) | 0) * bw;
      for (let x = 0; x < w; x++) {
        const v = gray[y * w + x];
        const b = row + ((x / B) | 0);
        if (v > peak[b]) peak[b] = v;
      }
    }
    const grid = new Float32Array(bw * bh);
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        let sum = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = by + dy;
            const xx = bx + dx;
            if (yy >= 0 && yy < bh && xx >= 0 && xx < bw) { sum += peak[yy * bw + xx]; n++; }
          }
        }
        grid[by * bw + bx] = Math.max(floor, sum / n);
      }
    }

    for (let y = 0, i = 0, k = 0; y < h; y++) {
      // Bilinear between block centres.
      const gy = Math.min(bh - 1, Math.max(0, (y + 0.5) / B - 0.5));
      const y0 = gy | 0;
      const y1 = Math.min(bh - 1, y0 + 1);
      const fy = gy - y0;
      for (let x = 0; x < w; x++, i++, k += 4) {
        const gx = Math.min(bw - 1, Math.max(0, (x + 0.5) / B - 0.5));
        const x0 = gx | 0;
        const x1 = Math.min(bw - 1, x0 + 1);
        const fx = gx - x0;
        const top = grid[y0 * bw + x0] + (grid[y0 * bw + x1] - grid[y0 * bw + x0]) * fx;
        const bot = grid[y1 * bw + x0] + (grid[y1 * bw + x1] - grid[y1 * bw + x0]) * fx;
        const back = top + (bot - top) * fy;
        let r = gray[i] / back;           // about 1 on paper, lower on ink
        r = (r - 0.55) / (0.9 - 0.55);    // stretch: 0.55 and below -> black, 0.9 and above -> white
        r = r < 0 ? 0 : r > 1 ? 1 : Math.pow(r, 1.4);
        p[k] = p[k + 1] = p[k + 2] = r * 255;
      }
    }
    const out = canvas(w, h);
    out.getContext('2d').putImageData(img, 0, 0);
    return out;
  }

  function toJpeg(c, quality) {
    return new Promise((resolve) => c.toBlob(resolve, 'image/jpeg', quality));
  }

  // ---- The corner editor --------------------------------------------------------

  const STYLE = `
  .rx { position: fixed; inset: 0; z-index: 50; background: #111418; color: #fff; display: flex; flex-direction: column;
    font: 16px/1.35 var(--font-body, system-ui, sans-serif); -webkit-user-select: none; user-select: none; }
  .rx-top { display: flex; align-items: center; gap: 10px; padding: calc(10px + env(safe-area-inset-top)) 14px 10px; }
  .rx-top b { flex: 1; font-size: 17px; }
  .rx-x { width: 44px; height: 44px; border: 0; border-radius: 8px; background: rgba(255,255,255,.12); color: #fff; font-size: 24px; line-height: 1; }
  .rx-stage { flex: 1; position: relative; touch-action: none; overflow: hidden; }
  .rx-stage canvas { position: absolute; }
  .rx-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
  .rx-loupe { position: absolute; top: 10px; width: 116px; height: 116px; border-radius: 50%; border: 3px solid #fff;
    box-shadow: 0 4px 16px rgba(0,0,0,.5); pointer-events: none; }
  .rx-msg { margin: 0; padding: 8px 16px 0; text-align: center; font-size: 14px; color: #C8CDD3; min-height: 27px; }
  .rx-bar { display: flex; gap: 8px; padding: 10px 14px calc(12px + env(safe-area-inset-bottom)); flex-wrap: wrap; justify-content: center; }
  .rx-bar button { min-height: 48px; padding: 0 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,.25);
    background: rgba(255,255,255,.1); color: #fff; font: inherit; font-weight: 600; }
  .rx-bar .rx-go { background: #fff; color: #111418; border-color: #fff; margin-left: auto; padding: 0 22px; }
  .rx-bar .rx-on { background: #fff; color: #111418; }
  .rx-preview { flex: 1; overflow: auto; display: grid; place-items: center; padding: 8px; }
  .rx-preview img { max-width: 100%; max-height: 100%; background: #fff; box-shadow: 0 4px 20px rgba(0,0,0,.5); }
  .rx-busy { position: absolute; inset: 0; display: grid; place-items: center; background: rgba(17,20,24,.55); font-weight: 600; }
  `;

  function el(html) {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstChild;
  }

  /**
   * Shows the photo with four corner handles on the receipt's edges, then a
   * preview of the straightened page. Resolves to {blob, url} or null.
   */
  async function scan(file) {
    if (!document.getElementById('rx-style')) {
      const s = document.createElement('style');
      s.id = 'rx-style';
      s.textContent = STYLE;
      document.head.appendChild(s);
    }
    const src = await loadImage(file);
    const root = el(`<div class="rx" role="dialog" aria-modal="true" aria-label="Adjust the receipt">
      <div class="rx-top"><b>Fit the corners to the receipt</b><button type="button" class="rx-x" aria-label="Cancel">&times;</button></div>
      <div class="rx-stage"><canvas></canvas><svg><polygon fill="rgba(80,160,255,.18)" stroke="#4DA3FF" stroke-width="2"/></svg>
        <canvas class="rx-loupe" width="232" height="232" hidden></canvas></div>
      <p class="rx-msg" aria-live="polite"></p>
      <div class="rx-bar"><button type="button" data-act="whole">Whole photo</button><button type="button" data-act="find">Find edges</button>
        <button type="button" class="rx-go" data-act="next">Next</button></div>
    </div>`);
    document.body.appendChild(root);
    document.documentElement.style.overflow = 'hidden';
    const stage = root.querySelector('.rx-stage');
    const view = stage.querySelector('canvas');
    const svg = stage.querySelector('svg');
    const poly = svg.querySelector('polygon');
    const loupe = stage.querySelector('.rx-loupe');
    const msg = root.querySelector('.rx-msg');
    let corners = inset(src, 0.04);
    let touched = false;
    let fit = { s: 1, ox: 0, oy: 0 };

    const handles = corners.map(() => {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.innerHTML = '<circle r="22" fill="transparent"/><circle r="11" fill="rgba(255,255,255,.25)" stroke="#fff" stroke-width="3"/>';
      svg.appendChild(g);
      return g;
    });

    function layout() {
      const W = stage.clientWidth;
      const H = stage.clientHeight;
      const pad = 22;
      const s = Math.min((W - pad * 2) / src.width, (H - pad * 2) / src.height);
      fit = { s: s, ox: (W - src.width * s) / 2, oy: (H - src.height * s) / 2 };
      const dpr = window.devicePixelRatio || 1;
      view.width = Math.round(src.width * s * dpr);
      view.height = Math.round(src.height * s * dpr);
      view.style.width = src.width * s + 'px';
      view.style.height = src.height * s + 'px';
      view.style.left = fit.ox + 'px';
      view.style.top = fit.oy + 'px';
      const ctx = view.getContext('2d');
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(src, 0, 0, view.width, view.height);
      draw();
    }
    const toScreen = (p) => ({ x: fit.ox + p.x * fit.s, y: fit.oy + p.y * fit.s });
    function draw() {
      const pts = corners.map(toScreen);
      poly.setAttribute('points', pts.map((p) => p.x + ',' + p.y).join(' '));
      handles.forEach((g, i) => g.setAttribute('transform', 'translate(' + pts[i].x + ',' + pts[i].y + ')'));
    }
    function showLoupe(p) {
      const ctx = loupe.getContext('2d');
      const r = 58 / 2.5 / fit.s; // photo pixels around the corner: 2.5 times the size on screen
      ctx.fillStyle = '#111418';
      ctx.fillRect(0, 0, 232, 232);
      ctx.drawImage(src, p.x - r, p.y - r, r * 2, r * 2, 0, 0, 232, 232);
      ctx.strokeStyle = '#4DA3FF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(116, 96); ctx.lineTo(116, 136); ctx.moveTo(96, 116); ctx.lineTo(136, 116);
      ctx.stroke();
      const sp = toScreen(p);
      loupe.style.left = sp.x < stage.clientWidth / 2 ? 'auto' : '10px';
      loupe.style.right = sp.x < stage.clientWidth / 2 ? '10px' : 'auto';
      loupe.hidden = false;
    }

    let dragging = -1;
    stage.addEventListener('pointerdown', (e) => {
      const r = stage.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      let best = -1;
      let bestD = 48;
      corners.forEach((c, i) => {
        const p = toScreen(c);
        const dd = Math.hypot(p.x - x, p.y - y);
        if (dd < bestD) { bestD = dd; best = i; }
      });
      if (best === -1) return;
      dragging = best;
      touched = true;
      stage.setPointerCapture(e.pointerId);
      showLoupe(corners[best]);
      e.preventDefault();
    });
    stage.addEventListener('pointermove', (e) => {
      if (dragging === -1) return;
      const r = stage.getBoundingClientRect();
      const x = (e.clientX - r.left - fit.ox) / fit.s;
      const y = (e.clientY - r.top - fit.oy) / fit.s;
      corners[dragging] = { x: Math.max(0, Math.min(src.width, x)), y: Math.max(0, Math.min(src.height, y)) };
      draw();
      showLoupe(corners[dragging]);
    });
    const stop = () => { dragging = -1; loupe.hidden = true; };
    stage.addEventListener('pointerup', stop);
    stage.addEventListener('pointercancel', stop);

    async function find(manual) {
      msg.textContent = 'Finding the receipt\u2019s edges\u2026';
      try {
        const found = await findCorners(src);
        if (touched && !manual) { msg.textContent = 'Drag each corner to a corner of the receipt.'; return; }
        if (found) {
          corners = found;
          draw();
          msg.textContent = 'Check the corners, then tap Next. Drag any that are off.';
        } else {
          msg.textContent = 'Couldn\u2019t find the edges. Drag each corner to a corner of the receipt.';
        }
      } catch (err) {
        msg.textContent = 'Drag each corner to a corner of the receipt.';
      }
    }

    window.addEventListener('resize', layout);
    layout();
    find(false);

    let previewing = false;
    const result = await new Promise((resolve) => {
      root.addEventListener('click', async (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        if (b.classList.contains('rx-x')) return resolve(null);
        if (b.dataset.act === 'whole') {
          corners = inset(src, 0);
          touched = true;
          draw();
          msg.textContent = 'Using the whole photo.';
        }
        if (b.dataset.act === 'find') find(true);
        if (b.dataset.act === 'next' && !previewing) {
          previewing = true;
          const page = await preview(root, src, corners);
          previewing = false;
          if (page) resolve(page);
        }
      });
    });
    window.removeEventListener('resize', layout);
    root.remove();
    document.documentElement.style.overflow = '';
    return result;
  }

  /** Second step: the straightened page, with Scan / Color and Rotate. Resolves to {blob, url} or null for Back. */
  async function preview(root, src, corners) {
    const busy = el('<div class="rx-busy">Straightening\u2026</div>');
    root.querySelector('.rx-stage').appendChild(busy);
    await new Promise((r) => setTimeout(r, 30));
    const flat = warp(src, corners);
    busy.remove();
    let scanned = null;
    let mode = 'scan';
    let turns = 0;

    const step = el(`<div class="rx" role="dialog" aria-modal="true" aria-label="Check the page">
      <div class="rx-top"><b>Check the page</b><button type="button" class="rx-x" aria-label="Back">&larr;</button></div>
      <div class="rx-preview"><img alt="The straightened receipt"></div>
      <p class="rx-msg">Make sure you can read the total and the date.</p>
      <div class="rx-bar"><button type="button" data-mode="scan">Scan</button><button type="button" data-mode="color">Color</button>
        <button type="button" data-act="rotate" aria-label="Rotate a quarter turn">&#8635; Rotate</button>
        <button type="button" class="rx-go" data-act="use">Use this page</button></div>
    </div>`);
    document.body.appendChild(step);
    const img = step.querySelector('img');
    let current;
    async function render() {
      if (mode === 'scan' && !scanned) scanned = scanLook(flat);
      current = rotate(mode === 'scan' ? scanned : flat, turns);
      const blob = await toJpeg(current, mode === 'scan' ? 0.7 : 0.78);
      if (img.src) URL.revokeObjectURL(img.src);
      img.src = URL.createObjectURL(blob);
      img.dataset.size = blob.size;
      step.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('rx-on', b.dataset.mode === mode));
      return blob;
    }
    // Taps can come before a page has finished drawing, so every button waits for the latest one.
    let rendering = render();
    return new Promise((resolve) => {
      step.addEventListener('click', async (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        if (b.classList.contains('rx-x')) {
          await rendering;
          URL.revokeObjectURL(img.src);
          step.remove();
          resolve(null);
          return;
        }
        if (b.dataset.mode) { mode = b.dataset.mode; rendering = rendering.then(render); }
        if (b.dataset.act === 'rotate') { turns++; rendering = rendering.then(render); }
        if (b.dataset.act === 'use') {
          const blob = await rendering;
          step.remove();
          resolve({ blob: blob, url: img.src, width: current.width, height: current.height });
        }
      });
    });
  }

  // ---- PDFs -----------------------------------------------------------------------

  /** A PDF file -> {bytes, pages}. Throws if it can't be read. */
  async function pdfInfo(file) {
    const PDFLib = await pdfLib();
    const bytes = new Uint8Array(await file.arrayBuffer());
    try {
      const doc = await PDFLib.PDFDocument.load(bytes, { ignoreEncryption: true });
      return { bytes: bytes, pages: doc.getPageCount() };
    } catch (err) {
      throw new Error('\u201c' + file.name + '\u201d couldn\u2019t be read as a PDF. Take a screenshot of it and add that instead.');
    }
  }

  /** [{jpeg: Blob} | {pdf: Uint8Array}] -> one PDF as base64 (no data: prefix). */
  async function buildPdf(items) {
    const PDFLib = await pdfLib();
    const doc = await PDFLib.PDFDocument.create();
    for (const item of items) {
      if (item.jpeg) {
        const image = await doc.embedJpg(new Uint8Array(await item.jpeg.arrayBuffer()));
        const margin = 18;
        const w = PAGE_WIDTH - margin * 2;
        const h = w * image.height / image.width;
        const page = doc.addPage([PAGE_WIDTH, h + margin * 2]);
        page.drawImage(image, { x: margin, y: margin, width: w, height: h });
      } else if (item.pdf) {
        const src = await PDFLib.PDFDocument.load(item.pdf, { ignoreEncryption: true });
        const pages = await doc.copyPages(src, src.getPageIndices());
        pages.forEach((p) => doc.addPage(p));
      }
    }
    doc.setTitle('Receipt');
    doc.setProducer('Purchase request page');
    return doc.saveAsBase64();
  }

  return {
    scan: scan,
    pdfInfo: pdfInfo,
    buildPdf: buildPdf,
    warm: () => { opencv().catch(() => {}); pdfLib().catch(() => {}); },
    // For testing in the browser console.
    _test: { loadImage, findCorners, warp, scanLook, order },
  };
})();

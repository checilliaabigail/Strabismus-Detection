/* ================================================================
   LANDMARK INDICES — MediaPipe Face Mesh (refineLandmarks: true)
   Total 478 landmarks when refineLandmarks is enabled.

   Corresponds to offline manual annotation scheme
   (Google Colaboratory web-based annotator):
     L_outer → 33   | R_outer → 263   (outer canthus)
     L_inner → 133  | R_inner → 362   (inner canthus)
     L_upper → 159  | R_upper → 386   (upper eyelid)
     L_lower → 145  | R_lower → 374   (lower eyelid)
     L_pupil → 468  | R_pupil → 473   (iris centre — needs refineLandmarks)
================================================================ */
const IDX = {
  L_outer: 33,  L_inner: 133, L_upper: 159, L_lower: 145, L_pupil: 468,
  R_outer: 263, R_inner: 362, R_upper: 386, R_lower: 374, R_pupil: 473,
};

/* ================================================================
   THRESHOLDS
   Derived from offline Gaussian log-likelihood (horizontal) and
   Gamma log-likelihood (vertical) analysis on 484 labelled
   cropped eye-region images.

   Horizontal (dx_norm) — 3 classes:
     T1 = -0.1519  (Exotropia | Normal boundary)
     T2 =  0.0484  (Normal | Esotropia boundary)
     Accuracy: 87.20% | Sensitivity: 86.24% | Specificity: 89.00%

   Vertical (|dy_norm|) — 2 classes:
     T  =  0.1160  (Normal | Vertical Strabismus boundary)
     Accuracy: 80.00% | Sensitivity: 79.49% | Specificity: 81.00%

   Vertical strabismus is not further subdivided into hypertropia
   or hypotropia because the geometry-based reference eye
   determination is insufficiently reliable for that distinction.
================================================================ */
const T1_DX = -0.1519;
const T2_DX =  0.0484;
const T_DY  =  0.1160;

/* ================================================================
   ROTATION HELPERS
   theta = atan2(R_outer.y - L_outer.y, R_outer.x - L_outer.x)
   All landmark coordinates are in normalised [0,1] space.
   Rotation centre = (0.5, 0.5) — image centre in normalised coords.
================================================================ */

function buildRotationMatrix(angle, cx, cy) {
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  return [
    [ cos, -sin, (1 - cos) * cx + sin * cy ],
    [ sin,  cos, -sin * cx + (1 - cos) * cy ],
  ];
}

function rotatePoint(px, py, M) {
  return {
    x: M[0][0] * px + M[0][1] * py + M[0][2],
    y: M[1][0] * px + M[1][1] * py + M[1][2],
  };
}

/* ================================================================
   MEDIAPIPE SETUP
================================================================ */
const faceMesh = new FaceMesh({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
faceMesh.onResults(onResults);

/* ================================================================
   DOM REFS
================================================================ */
const imageInput        = document.getElementById('imageInput');
const uploadZone        = document.getElementById('uploadZone');
const previewSection    = document.getElementById('preview-section');
const previewCanvas     = document.getElementById('previewCanvas');
const processingOverlay = document.getElementById('processingOverlay');
const resultsSection    = document.getElementById('results-section');
const errorBanner       = document.getElementById('errorBanner');
const resetBtn          = document.getElementById('resetBtn');
let   originalImage     = null;

/* ================================================================
   IMAGE UPLOAD
================================================================ */
imageInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    originalImage = img;
    previewCanvas.width  = img.naturalWidth;
    previewCanvas.height = img.naturalHeight;
    previewCanvas.getContext('2d').drawImage(img, 0, 0);
    uploadZone.style.display     = 'none';
    previewSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorBanner.classList.remove('show');
    processingOverlay.classList.add('show');
    await faceMesh.send({ image: img });
  };
  img.src = URL.createObjectURL(file);
});

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('active'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('active'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const dt = new DataTransfer(); dt.items.add(file);
    imageInput.files = dt.files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

resetBtn.addEventListener('click', () => {
  uploadZone.style.display     = 'block';
  previewSection.style.display = 'none';
  resultsSection.style.display = 'none';
  errorBanner.classList.remove('show');
  imageInput.value = '';
  originalImage    = null;
});

/* ================================================================
   FACE MESH CALLBACK
================================================================ */
function onResults(results) {
  processingOverlay.classList.remove('show');

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    showError('No face detected. Please upload a clear frontal facial image.');
    return;
  }

  const lm = results.multiFaceLandmarks[0];
  if (lm.length < 478) {
    showError('Iris landmarks unavailable. Ensure both eyes are fully visible.');
    return;
  }

  // Step 1: Extract raw normalised coordinates
  const raw = {
    L_outer: lm[IDX.L_outer], L_inner: lm[IDX.L_inner],
    L_upper: lm[IDX.L_upper], L_lower: lm[IDX.L_lower],
    L_pupil: lm[IDX.L_pupil],
    R_outer: lm[IDX.R_outer], R_inner: lm[IDX.R_inner],
    R_upper: lm[IDX.R_upper], R_lower: lm[IDX.R_lower],
    R_pupil: lm[IDX.R_pupil],
  };

  // Step 2: Head tilt angle θ
  const theta    = Math.atan2(
    raw.R_outer.y - raw.L_outer.y,
    raw.R_outer.x - raw.L_outer.x
  );
  const thetaDeg = theta * 180 / Math.PI;

  // Step 3: Rotation correction
  const M = buildRotationMatrix(theta, 0.5, 0.5);
  const r = {};
  for (const key of Object.keys(raw)) {
    r[key] = rotatePoint(raw[key].x, raw[key].y, M);
  }

  // Step 4: dx_norm
  const eye_width_L  = r.L_inner.x - r.L_outer.x;
  const eye_width_R  = r.R_outer.x - r.R_inner.x;
  const left_norm_x  = (r.L_pupil.x - r.L_outer.x) / eye_width_L;
  const right_norm_x = (r.R_pupil.x - r.R_inner.x) / eye_width_R;
  const dx_norm      = left_norm_x - right_norm_x;

  // Step 5: dy_norm
  const dist_L  = Math.abs(r.L_pupil.y - r.L_outer.y);
  const dist_R  = Math.abs(r.R_pupil.y - r.R_outer.y);
  const dy      = dist_L < dist_R
                    ? r.R_pupil.y - r.L_pupil.y
                    : r.L_pupil.y - r.R_pupil.y;
  const y_ref   = ((r.L_lower.y - r.L_upper.y) + (r.R_lower.y - r.R_upper.y)) / 2;
  const dy_norm = dy / y_ref;
  const abs_dy  = Math.abs(dy_norm);

  // Step 6: Horizontal classification
  let hClass, hIcon, hDesc, hCardCls, hResCls;
  if (dx_norm < T1_DX) {
    hClass = 'EXOTROPIA'; hIcon = '👁️';
    hDesc  = 'Outward horizontal deviation detected (temporal deviation).';
    hCardCls = 'exo'; hResCls = 'result-exo';
  } else if (dx_norm <= T2_DX) {
    hClass = 'NORMAL'; hIcon = '✅';
    hDesc  = 'No significant horizontal deviation detected.';
    hCardCls = 'norm'; hResCls = 'result-norm';
  } else {
    hClass = 'ESOTROPIA'; hIcon = '👁️';
    hDesc  = 'Inward horizontal deviation detected (nasal deviation).';
    hCardCls = 'eso'; hResCls = 'result-eso';
  }

  // Step 7: Vertical classification
  let vClass, vIcon, vDesc, vCardCls, vResCls;
  if (abs_dy <= T_DY) {
    vClass = 'NORMAL'; vIcon = '✅';
    vDesc  = 'No significant vertical deviation detected.';
    vCardCls = 'norm'; vResCls = 'result-norm';
  } else {
    vClass = 'VERTICAL STRABISMUS'; vIcon = '↕️';
    vDesc  = 'Vertical deviation detected (hypertropia or hypotropia).';
    vCardCls = 'vert'; vResCls = 'result-vert';
  }

  // Step 8: Draw landmarks on canvas
  const ctx = previewCanvas.getContext('2d');
  const W   = previewCanvas.width;
  const H   = previewCanvas.height;

  if (originalImage) ctx.drawImage(originalImage, 0, 0);

  const rawPts   = [raw.L_outer, raw.L_inner, raw.L_upper, raw.L_lower, raw.L_pupil,
                    raw.R_outer, raw.R_inner, raw.R_upper, raw.R_lower, raw.R_pupil];
  const ptColors = ['#4fffb0','#4fffb0','#4fffb0','#4fffb0','#ff6b6b',
                    '#6b9fff','#6b9fff','#6b9fff','#6b9fff','#ff6b6b'];
  const ptLabels = ['L_out','L_in','L_up','L_lo','L_pup',
                    'R_out','R_in','R_up','R_lo','R_pup'];

  rawPts.forEach((pt, i) => {
    const x = pt.x * W, y = pt.y * H;
    ctx.beginPath();
    ctx.arc(x, y, i === 4 || i === 9 ? 5 : 3, 0, 2 * Math.PI);
    ctx.fillStyle = ptColors[i];
    ctx.fill();
    ctx.font = 'bold 10px DM Mono, monospace';
    ctx.fillStyle = ptColors[i];
    ctx.fillText(ptLabels[i], x + 6, y - 4);
  });

  ctx.strokeStyle = 'rgba(79,255,176,0.5)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(raw.L_outer.x * W, raw.L_outer.y * H);
  ctx.lineTo(raw.L_inner.x * W, raw.L_inner.y * H);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(raw.R_inner.x * W, raw.R_inner.y * H);
  ctx.lineTo(raw.R_outer.x * W, raw.R_outer.y * H);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,107,107,0.6)';
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(raw.L_pupil.x * W, raw.L_pupil.y * H);
  ctx.lineTo(raw.R_pupil.x * W, raw.R_pupil.y * H);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = 'rgba(255,209,102,0.8)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(raw.L_outer.x * W, raw.L_outer.y * H, 22, 0, -theta, theta >= 0);
  ctx.stroke();

  // Step 9: Update UI
  document.getElementById('classifyCard').className    = `classify-card ${hCardCls}`;
  document.getElementById('classifyIcon').textContent   = hIcon;
  document.getElementById('classifyResult').textContent = hClass;
  document.getElementById('classifyResult').className   = `classify-result ${hResCls}`;
  document.getElementById('classifyDesc').textContent   = hDesc;

  document.getElementById('classifyCardV').className    = `classify-card ${vCardCls}`;
  document.getElementById('classifyIconV').textContent   = vIcon;
  document.getElementById('classifyResultV').textContent = vClass;
  document.getElementById('classifyResultV').className   = `classify-result ${vResCls}`;
  document.getElementById('classifyDescV').textContent   = vDesc;

  document.getElementById('thetaVal').textContent  = `θ = ${thetaDeg.toFixed(3)}°`;
  document.getElementById('leftVal').textContent   = left_norm_x.toFixed(4);
  document.getElementById('rightVal').textContent  = right_norm_x.toFixed(4);
  document.getElementById('dxVal').textContent     = dx_norm.toFixed(4);
  document.getElementById('dyRaw').textContent     = dy.toFixed(4);
  document.getElementById('yRef').textContent      = y_ref.toFixed(4);
  document.getElementById('dyVal').textContent     = abs_dy.toFixed(4);

  document.getElementById('lmLeft').innerHTML =
    `outer: (${r.L_outer.x.toFixed(4)}, ${r.L_outer.y.toFixed(4)})<br>` +
    `inner: (${r.L_inner.x.toFixed(4)}, ${r.L_inner.y.toFixed(4)})<br>` +
    `upper: (${r.L_upper.x.toFixed(4)}, ${r.L_upper.y.toFixed(4)})<br>` +
    `lower: (${r.L_lower.x.toFixed(4)}, ${r.L_lower.y.toFixed(4)})<br>` +
    `pupil: (${r.L_pupil.x.toFixed(4)}, ${r.L_pupil.y.toFixed(4)})`;

  document.getElementById('lmRight').innerHTML =
    `outer: (${r.R_outer.x.toFixed(4)}, ${r.R_outer.y.toFixed(4)})<br>` +
    `inner: (${r.R_inner.x.toFixed(4)}, ${r.R_inner.y.toFixed(4)})<br>` +
    `upper: (${r.R_upper.x.toFixed(4)}, ${r.R_upper.y.toFixed(4)})<br>` +
    `lower: (${r.R_lower.x.toFixed(4)}, ${r.R_lower.y.toFixed(4)})<br>` +
    `pupil: (${r.R_pupil.x.toFixed(4)}, ${r.R_pupil.y.toFixed(4)})`;

  resultsSection.style.display = 'block';
}

/* ================================================================
   HELPERS
================================================================ */
function showError(msg) {
  errorBanner.textContent = '⚠ ' + msg;
  errorBanner.classList.add('show');
  resultsSection.style.display = 'none';
}

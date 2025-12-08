let imgElement = document.getElementById("filePreview");
let fileInput = document.getElementById("fileInput");
let analyzeBtn = document.getElementById("analyzeBtn");
let loading = document.getElementById("loading");
let results = document.getElementById("results");
let resultContent = document.getElementById("resultContent");

let src, gray;
let faceCascade, eyeCascade;

fileInput.addEventListener("change", function () {
    let file = fileInput.files[0];

    if (!file) return;

    document.getElementById("fileInfo").textContent = file.name;

    imgElement.src = URL.createObjectURL(file);
    imgElement.style.display = "block";

    analyzeBtn.disabled = false;
});

function loadCascades() {
    return new Promise(resolve => {
        faceCascade = new cv.CascadeClassifier();
        eyeCascade = new cv.CascadeClassifier();

        faceCascade.load("haarcascade_frontalface_default.xml");
        eyeCascade.load("haarcascade_eye.xml");

        resolve();
    });
}

async function analyzeImage() {
    analyzeBtn.disabled = true;
    loading.style.display = "block";
    results.style.display = "none";

    await loadCascades();

    src = cv.imread("filePreview");
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // FACE DETECTION
    let faces = new cv.RectVector();
    let faceMat = new cv.Mat();

    cv.equalizeHist(gray, faceMat);
    faceCascade.detectMultiScale(faceMat, faces, 1.1, 3, 0);

    if (faces.size() === 0) {
        showResult("❌ Wajah tidak terdeteksi");
        cleanup();
        return;
    }

    let face = faces.get(0);
    let faceROI = gray.roi(face);

    // EYE DETECTION
    let eyes = new cv.RectVector();
    eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0);

    if (eyes.size() < 2) {
        showResult("❌ Tidak cukup mata terdeteksi");
        cleanup();
        return;
    }

    // Sort eyes by x-position
    let eyeList = [];
    for (let i = 0; i < eyes.size(); i++) {
        let e = eyes.get(i);
        eyeList.push({ x: e.x, y: e.y, w: e.width, h: e.height });
    }
    eyeList.sort((a, b) => a.x - b.x);

    let leftEye = faceROI.roi(new cv.Rect(eyeList[0].x, eyeList[0].y, eyeList[0].w, eyeList[0].h));
    let rightEye = faceROI.roi(new cv.Rect(eyeList[1].x, eyeList[1].y, eyeList[1].w, eyeList[1].h));

    let leftCircle = detectPupil(leftEye);
    let rightCircle = detectPupil(rightEye);

    if (!leftCircle || !rightCircle) {
        showResult("❌ Pupil tidak terdeteksi");
        cleanup();
        return;
    }

    // NORMALIZED POSITION
    let leftNorm = leftCircle.x / leftEye.cols;
    let rightNorm = rightCircle.x / rightEye.cols;
    let dx = leftNorm - rightNorm;

    showResult(`
        <div class="result-item"><b>Left Pupil:</b> (${leftCircle.x}, ${leftCircle.y}), r=${leftCircle.r}</div>
        <div class="result-item"><b>Right Pupil:</b> (${rightCircle.x}, ${rightCircle.y}), r=${rightCircle.r}</div>
        <div class="result-item"><b>Left Norm:</b> ${leftNorm.toFixed(3)}</div>
        <div class="result-item"><b>Right Norm:</b> ${rightNorm.toFixed(3)}</div>
        <div class="result-item"><b>dx:</b> ${dx.toFixed(3)}</div>
    `);

    cleanup();
}

function detectPupil(eyeMat) {
    let circles = new cv.Mat();
    cv.HoughCircles(
        eyeMat,
        circles,
        cv.HOUGH_GRADIENT,
        1,
        20,
        100,
        15,
        5,
        40
    );

    if (circles.rows === 0) return null;

    let x = circles.data32F[0];
    let y = circles.data32F[1];
    let r = circles.data32F[2];

    return { x, y, r };
}

function showResult(html) {
    loading.style.display = "none";
    results.style.display = "block";
    resultContent.innerHTML = html;
}

function cleanup() {
    if (src) src.delete();
    if (gray) gray.delete();
}

document.getElementById("analyzeBtn").addEventListener("click", analyzeImage);

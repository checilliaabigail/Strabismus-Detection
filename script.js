let faceCascade, eyeCascade;
let cvReady = false;

// Dipanggil otomatis setelah opencv.js selesai load
function onOpenCvReady() {
    console.log("OpenCV is ready!");
    cvReady = true;

    // load cascade setelah OpenCV siap
    loadCascadeFiles();
}

// Load XML cascade files
function loadCascadeFiles() {
    loadCascade("haarcascade_frontalface_default.xml", (classifier) => {
        faceCascade = classifier;
        console.log("Face cascade loaded");
    });

    loadCascade("haarcascade_eye.xml", (classifier) => {
        eyeCascade = classifier;
        console.log("Eye cascade loaded");
    });
}

function loadCascade(filename, callback) {
    let xhr = new XMLHttpRequest();
    xhr.open("GET", filename, true);
    xhr.responseType = "arraybuffer";

    xhr.onload = () => {
        if (xhr.status === 200) {
            let data = new Uint8Array(xhr.response);
            cv.FS_createDataFile("/", filename, data, true, false, false);

            let classifier = new cv.CascadeClassifier();
            classifier.load(filename);

            callback(classifier);
        } else {
            console.error("Failed to load:", filename);
        }
    };

    xhr.send();
}

// ===== DOM ======
const fileInput = document.getElementById("fileInput");
const filePreview = document.getElementById("filePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingDiv = document.getElementById("loading");
const resultBox = document.getElementById("results");
const resultContent = document.getElementById("resultContent");

// Show preview
fileInput.addEventListener("change", () => {
    let file = fileInput.files[0];
    if (!file) return;

    filePreview.src = URL.createObjectURL(file);
    filePreview.style.display = "block";
    analyzeBtn.disabled = false;
});

// ANALYZE
analyzeBtn.addEventListener("click", () => {
    if (!cvReady) {
        alert("OpenCV belum siap. Tunggu 1-2 detik lalu coba lagi.");
        return;
    }

    if (!faceCascade || !eyeCascade) {
        alert("Cascade belum siap.");
        return;
    }

    loadingDiv.style.display = "block";
    resultBox.style.display = "none";

    setTimeout(runAnalysis, 200);
});

// ===== Main Logic =====
function runAnalysis() {
    let img = filePreview;

    let src = cv.imread(img);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // FACE DETECTION
    let faces = new cv.RectVector();
    let msize = new cv.Size(80,80);
    faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, msize);

    if (faces.size() === 0) {
        showResult("❌ Wajah tidak ditemukan.");
        cleanup(src, gray);
        return;
    }

    let face = faces.get(0);
    let faceROI = gray.roi(face);

    // EYE DETECTION
    let eyes = new cv.RectVector();
    eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0);

    if (eyes.size() < 2) {
        showResult("❌ Mata tidak ditemukan.");
        cleanup(src, gray, faceROI);
        return;
    }

    let eyeRects = [];
    for (let i = 0; i < eyes.size(); i++) {
        eyeRects.push(eyes.get(i));
    }
    eyeRects.sort((a,b)=>a.x - b.x);

    let leftROI = faceROI.roi(eyeRects[0]);
    let rightROI = faceROI.roi(eyeRects[1]);

    let leftPupil = detectPupilHoughFast(leftROI);
    let rightPupil = detectPupilHoughFast(rightROI);

    if (!leftPupil || !rightPupil) {
        showResult("❌ Pupil tidak ditemukan.");
        cleanup(src, gray, faceROI, leftROI, rightROI);
        return;
    }

    let leftNorm = leftPupil.x / leftROI.cols;
    let rightNorm = rightPupil.x / rightROI.cols;
    let dx = rightNorm - leftNorm;

    let html = `
        <div class="result-item"><b>Left Pupil:</b> ${JSON.stringify(leftPupil)}</div>
        <div class="result-item"><b>Right Pupil:</b> ${JSON.stringify(rightPupil)}</div>
        <div class="result-item"><b>Left Normalized X:</b> ${leftNorm.toFixed(3)}</div>
        <div class="result-item"><b>Right Normalized X:</b> ${rightNorm.toFixed(3)}</div>
        <div class="result-item"><b>DX:</b> ${dx.toFixed(3)}</div>
    `;

    showResult(html);

    cleanup(src, gray, faceROI, leftROI, rightROI);
}

// ===== Hough Circle Pupil Detection =====
function detectPupilHoughFast(eyeMat) {
    let eq = new cv.Mat();
    cv.equalizeHist(eyeMat, eq);

    let circles = new cv.Mat();
    cv.HoughCircles(eq, circles, cv.HOUGH_GRADIENT,
        1.2, 20, 80, 15, 5, 40);

    if (circles.rows === 0) {
        eq.delete(); circles.delete();
        return null;
    }

    let x = circles.data32F[0];
    let y = circles.data32F[1];
    let r = circles.data32F[2];

    eq.delete(); circles.delete();

    return {x: Math.round(x), y: Math.round(y), r: Math.round(r)};
}

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => m && m.delete());
}

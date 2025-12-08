let faceCascade, eyeCascade;
let cvReady = false;

function onOpenCvReady() {
    console.log("OpenCV is ready!");
    cvReady = true;

    // Load cascade files
    loadCascade("haarcascade_frontalface_default.xml", (classifier) => {
        faceCascade = classifier;
        console.log("Face cascade loaded");
    });

    loadCascade("haarcascade_eye.xml", (classifier) => {
        eyeCascade = classifier;
        console.log("Eye cascade loaded");
    });
}

// Load XML cascade
function loadCascade(filename, callback) {
    let xhr = new XMLHttpRequest();
    xhr.open("GET", filename, true);
    xhr.responseType = "arraybuffer";
    xhr.onload = function () {
        if (xhr.status === 200) {
            let data = new Uint8Array(xhr.response);
            cv.FS_createDataFile("/", filename, data, true, false, false);
            let classifier = new cv.CascadeClassifier();
            classifier.load(filename);
            callback(classifier);
        } else {
            console.error("Failed to load cascade:", filename);
        }
    };
    xhr.send();
}

// ------- DOM elements -------
const fileInput = document.getElementById("fileInput");
const filePreview = document.getElementById("filePreview");
const fileInfo = document.getElementById("fileInfo");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingDiv = document.getElementById("loading");
const resultBox = document.getElementById("results");
const resultContent = document.getElementById("resultContent");

let uploadedImage = null;

// Upload preview
fileInput.addEventListener("change", function () {
    let file = this.files[0];
    if (!file) return;

    filePreview.src = URL.createObjectURL(file);
    filePreview.style.display = "block";
    fileInfo.textContent = file.name;

    analyzeBtn.disabled = false;
});

// Main ANALYZE button
analyzeBtn.addEventListener("click", async () => {
    if (!cvReady) {
        alert("OpenCV belum siap!");
        return;
    }

    loadingDiv.style.display = "block";
    resultBox.style.display = "none";

    setTimeout(runAnalysis, 300);
});

// -------- MAIN ANALYSIS ---------

function runAnalysis() {
    let imgElement = filePreview;
    let src = cv.imread(imgElement);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // -------- FACE DETECTION --------
    let faces = new cv.RectVector();
    let msize = new cv.Size(80, 80);
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, msize);

    if (faces.size() === 0) {
        showResult("❌ Wajah tidak ditemukan.");
        cleanup(src, gray);
        return;
    }

    // Ambil wajah terbesar
    let face = faces.get(0);
    let faceROI = gray.roi(face);

    // -------- EYE DETECTION --------
    let eyes = new cv.RectVector();
    eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0);

    if (eyes.size() < 2) {
        showResult("❌ Mata tidak terdeteksi dengan cukup.");
        cleanup(src, gray, faceROI);
        return;
    }

    // Sort by X position → left then right
    let eyeList = [];
    for (let i = 0; i < eyes.size(); i++) {
        let e = eyes.get(i);
        eyeList.push({x:e.x, y:e.y, w:e.width, h:e.height});
    }
    eyeList.sort((a,b) => a.x - b.x);

    let leftEyeROI = faceROI.roi(new cv.Rect(eyeList[0].x, eyeList[0].y, eyeList[0].w, eyeList[0].h));
    let rightEyeROI = faceROI.roi(new cv.Rect(eyeList[1].x, eyeList[1].y, eyeList[1].w, eyeList[1].h));

    // -------- PUPIL DETECTION (HOUGH FAST) --------
    let leftPupil = detectPupilHoughFast(leftEyeROI);
    let rightPupil = detectPupilHoughFast(rightEyeROI);

    if (!leftPupil || !rightPupil) {
        showResult("❌ Pupil tidak terdeteksi pada salah satu mata.");
        cleanup(src, gray, faceROI, leftEyeROI, rightEyeROI);
        return;
    }

    // -------- COMPUTE DX --------
    let leftNorm = leftPupil.x / leftEyeROI.cols;
    let rightNorm = rightPupil.x / rightEyeROI.cols;
    let dx = leftNorm - rightNorm;

    let html = `
        <div class="result-item"><b>Left Pupil:</b> (${leftPupil.x}, ${leftPupil.y}), r=${leftPupil.r}</div>
        <div class="result-item"><b>Right Pupil:</b> (${rightPupil.x}, ${rightPupil.y}), r=${rightPupil.r}</div>
        <div class="result-item"><b>Left Normalized X:</b> ${leftNorm.toFixed(3)}</div>
        <div class="result-item"><b>Right Normalized X:</b> ${rightNorm.toFixed(3)}</div>
        <div class="result-item"><b>DX:</b> ${dx.toFixed(3)}</div>
    `;

    showResult(html);

    cleanup(src, gray, faceROI, leftEyeROI, rightEyeROI);
}

// -------- HOUGH PUPIL DETECTOR (FAST VERSION) --------

function detectPupilHoughFast(eyeMat) {
    let equalized = new cv.Mat();
    cv.equalizeHist(eyeMat, equalized);

    let circles = new cv.Mat();
    cv.HoughCircles(
        equalized,
        circles,
        cv.HOUGH_GRADIENT,
        1.2,      // dp
        20,       // minDist
        80,       // param1
        15,       // param2
        5,        // minRadius
        40        // maxRadius
    );

    if (circles.rows === 0) {
        equalized.delete();
        circles.delete();
        return null;
    }

    let x = circles.data32F[0];
    let y = circles.data32F[1];
    let r = circles.data32F[2];

    equalized.delete();
    circles.delete();

    return {x: Math.round(x), y: Math.round(y), r: Math.round(r)};
}

// -------- Utility --------

function cleanup(...mats) {
    mats.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
}

function showResult(html) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = html;
}

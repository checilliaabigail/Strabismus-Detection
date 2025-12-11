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
    let dx = leftNorm - rightNorm;

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

// ===== Hough Circle Pupil Detection (Auto - Multiple Parameters) =====
function detectPupilHoughFast(eyeMat) {
    // Apply Gaussian blur
    let blurred = new cv.Mat();
    let ksize = new cv.Size(7, 7);
    cv.GaussianBlur(eyeMat, blurred, ksize, 2, 2, cv.BORDER_DEFAULT);

    // Parameter sets (sama seperti Python)
    const paramSets = [
        // Set 1: Parameter normal
        {dp: 1.1, param1: 80, param2: 20, minRadius: 15, maxRadius: 40},
        // Set 2: Lebih sensitif
        {dp: 1.0, param1: 60, param2: 15, minRadius: 10, maxRadius: 45},
        // Set 3: Sangat sensitif (fallback)
        {dp: 0.9, param1: 40, param2: 10, minRadius: 5, maxRadius: 50},
        // Set 4: Untuk pupil besar
        {dp: 1.2, param1: 100, param2: 25, minRadius: 20, maxRadius: 35},
        // Set 5: Untuk pupil kecil
        {dp: 1.0, param1: 90, param2: 30, minRadius: 8, maxRadius: 30}
    ];

    let allCircles = [];
    let eyeWidth = eyeMat.cols;
    let eyeHeight = eyeMat.rows;

    // Try all parameter sets
    for (let params of paramSets) {
        let circles = new cv.Mat();
        try {
            cv.HoughCircles(
                blurred, 
                circles, 
                cv.HOUGH_GRADIENT,
                params.dp,           // dp
                20,                  // minDist
                params.param1,       // param1
                params.param2,       // param2
                params.minRadius,    // minRadius
                params.maxRadius     // maxRadius
            );

            // Process detected circles
            if (circles.cols > 0) {
                for (let i = 0; i < circles.cols; i++) {
                    let x = circles.data32F[i * 3];
                    let y = circles.data32F[i * 3 + 1];
                    let r = circles.data32F[i * 3 + 2];

                    // Validasi dasar
                    if (x - r > 0 && x + r < eyeWidth &&
                        y - r > 0 && y + r < eyeHeight &&
                        r >= 5 && r <= 50) {
                        allCircles.push({x: x, y: y, r: r});
                    }
                }
            }
        } catch(e) {
            console.log("Error in param set:", params, e);
        }
        circles.delete();
    }

    blurred.delete();

    // Select best circle based on scoring
    if (allCircles.length === 0) {
        return null;
    }

    let centerX = eyeWidth / 2;
    let centerY = eyeHeight / 2;
    let maxDim = Math.max(eyeWidth, eyeHeight);

    let scoredCircles = allCircles.map(circle => {
        // Distance to center
        let distance = Math.sqrt(
            Math.pow(circle.x - centerX, 2) + 
            Math.pow(circle.y - centerY, 2)
        );

        // Center score
        let centerScore = 1.0 / (1.0 + distance / maxDim);

        // Radius score (prefer 15-35 pixels)
        let radiusScore;
        if (circle.r >= 15 && circle.r <= 35) {
            radiusScore = 1.0;
        } else if (circle.r >= 10 && circle.r <= 40) {
            radiusScore = 0.7;
        } else {
            radiusScore = 0.3;
        }

        let totalScore = centerScore * 0.7 + radiusScore * 0.3;

        return {
            circle: circle,
            score: totalScore
        };
    });

    // Sort by score (highest first)
    scoredCircles.sort((a, b) => b.score - a.score);

    let bestCircle = scoredCircles[0].circle;

    return {
        x: Math.round(bestCircle.x), 
        y: Math.round(bestCircle.y), 
        r: Math.round(bestCircle.r)
    };
}

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => m && m.delete());
}

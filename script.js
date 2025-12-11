let faceCascade, eyeCascade;
let cvReady = false;
let cascadesReady = false;

// Dipanggil otomatis setelah opencv.js selesai load
function onOpenCvReady() {
    console.log("OpenCV loading...");
    
    // Tunggu cv module benar-benar siap (synchronous check)
    let checkInterval = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat && cv.CascadeClassifier) {
            clearInterval(checkInterval);
            console.log("✅ OpenCV is ready!");
            cvReady = true;
            
            // Load cascades SETELAH OpenCV siap
            loadCascadeFilesSequential();
        }
    }, 100);
}

// Load cascade files secara sequential (synchronous)
function loadCascadeFilesSequential() {
    console.log("Loading Haar Cascades...");
    
    // Load face cascade dulu
    loadCascade("haarcascade_frontalface_default.xml", (classifier) => {
        faceCascade = classifier;
        console.log("✅ Face cascade loaded");
        
        // Setelah face cascade selesai, baru load eye cascade
        loadCascade("haarcascade_eye.xml", (classifier) => {
            eyeCascade = classifier;
            console.log("✅ Eye cascade loaded");
            
            // Semua cascade sudah siap
            cascadesReady = true;
            console.log("✅ All cascades ready!");
        });
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

// ===== DOM =====
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

// ===== Detect Face (SAMA SEPERTI PYTHON) =====
function detectFace(gray) {
    let faces = new cv.RectVector();
    let minSize = new cv.Size(80, 80);
    
    // detectMultiScale dengan parameter SAMA seperti Python
    faceCascade.detectMultiScale(
        gray,
        faces,
        1.1,        // scaleFactor
        5,          // minNeighbors
        0,          // flags
        minSize     // minSize (80, 80)
    );

    if (faces.size() === 0) {
        console.log("❌ Tidak ada wajah terdeteksi");
        faces.delete();
        return null;
    }

    // Ambil wajah dengan area terbesar (paling umum)
    let facesList = [];
    for (let i = 0; i < faces.size(); i++) {
        let face = faces.get(i);
        facesList.push({
            x: face.x,
            y: face.y,
            width: face.width,
            height: face.height,
            area: face.width * face.height
        });
    }

    // Sort by area (largest first)
    facesList.sort((a, b) => b.area - a.area);
    let largestFace = facesList[0];

    // Crop wajah (face ROI)
    let faceGray = gray.roi(new cv.Rect(largestFace.x, largestFace.y, largestFace.width, largestFace.height));

    faces.delete();

    return {
        faceBox: largestFace,
        faceGray: faceGray
    };
}

// ===== Improved Eye Detection (100% SAMA SEPERTI PYTHON) =====
function improvedEyeDetection(gray) {
    /**
     * Deteksi mata yang lebih baik untuk gambar close-up mata
     * PERSIS seperti Python version
     */
    
    // Parameter yang lebih sensitif untuk gambar close-up
    const scaleFactors = [1.01, 1.02, 1.05, 1.1];
    const minNeighborsList = [1, 2, 3];
    const minSizes = [
        new cv.Size(10, 10),
        new cv.Size(15, 15),
        new cv.Size(20, 20),
        new cv.Size(25, 25)
    ];

    let bestEyes = null;
    let bestScore = -1;
    let bestParams = null;

    for (let sf of scaleFactors) {
        for (let mn of minNeighborsList) {
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                
                try {
                    eyeCascade.detectMultiScale(
                        gray,
                        eyes,
                        sf,      // scaleFactor
                        mn,      // minNeighbors
                        0,       // flags (cv2.CASCADE_SCALE_IMAGE = 0)
                        ms       // minSize
                    );

                    // Filter: hanya ambil 2 mata dengan area terbesar
                    if (eyes.size() >= 2) {
                        // areas = [(ex, ey, ew, eh, ew*eh) for (ex, ey, ew, eh) in eyes]
                        let areas = [];
                        for (let i = 0; i < eyes.size(); i++) {
                            let eye = eyes.get(i);
                            areas.push({
                                x: eye.x,
                                y: eye.y,
                                width: eye.width,
                                height: eye.height,
                                area: eye.width * eye.height
                            });
                        }

                        // areas_sorted = sorted(areas, key=lambda x: x[4], reverse=True)[:2]
                        areas.sort((a, b) => b.area - a.area);
                        let eye1 = areas[0];
                        let eye2 = areas[1];

                        // Pastikan kedua mata tidak tumpang tindih
                        let x1 = eye1.x, y1 = eye1.y, w1 = eye1.width, h1 = eye1.height;
                        let x2 = eye2.x, y2 = eye2.y, w2 = eye2.width, h2 = eye2.height;

                        // Hitung overlap
                        let dx = Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2);
                        let dy = Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2);

                        if (dx <= 0 || dy <= 0) {  // Tidak overlap
                            let score = eye1.area + eye2.area;
                            if (score > bestScore) {
                                bestScore = score;
                                // Convert to regular integers immediately
                                bestEyes = [
                                    {x: x1, y: y1, width: w1, height: h1},
                                    {x: x2, y: y2, width: w2, height: h2}
                                ];
                                bestParams = {sf: sf, mn: mn, minSize: ms.width};
                            }
                        }
                    }
                } catch(e) {
                    console.log("Error in eye detection params:", e);
                }
                
                eyes.delete();
            }
        }
    }

    // return (best_eyes if best_eyes else [], best_params)
    return {
        bestEyes: bestEyes ? bestEyes : [],
        bestParams: bestParams
    };
}

// ===== Main Logic =====
function runAnalysis() {
    let img = filePreview;

    let src = cv.imread(img);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // FACE DETECTION (SAMA SEPERTI PYTHON)
    let faceResult = detectFace(gray);
    
    if (!faceResult) {
        showResult("❌ Wajah tidak ditemukan.");
        cleanup(src, gray);
        return;
    }

    let {faceBox, faceGray} = faceResult;
    console.log("Face detected:", faceBox);

    // EYE DETECTION DI DALAM WAJAH (SAMA SEPERTI PYTHON)
    let eyeResult = improvedEyeDetection(faceGray);
    
    if (!eyeResult.bestEyes || eyeResult.bestEyes.length < 2) {
        showResult("❌ Mata tidak ditemukan.");
        cleanup(src, gray, faceGray);
        return;
    }

    let eyesInFace = eyeResult.bestEyes;
    console.log("Eyes dalam face:", eyesInFace);
    console.log("Best params:", eyeResult.bestParams);

    // Sort by x position (kiri ke kanan)
    let eyes = eyesInFace.sort((a, b) => a.x - b.x);
    console.log("Final eyes (sorted):", eyes);

    // Extract left and right eye ROI
    let leftEyeRect = eyes[0];
    let rightEyeRect = eyes[1];

    // PENTING: Clone faceGray untuk setiap eye ROI agar tidak corrupt
    let leftROI = faceGray.roi(new cv.Rect(leftEyeRect.x, leftEyeRect.y, leftEyeRect.width, leftEyeRect.height)).clone();
    let rightROI = faceGray.roi(new cv.Rect(rightEyeRect.x, rightEyeRect.y, rightEyeRect.width, rightEyeRect.height)).clone();

    console.log("Left Eye ROI size:", leftROI.cols, "x", leftROI.rows);
    console.log("Right Eye ROI size:", rightROI.cols, "x", rightROI.rows);

    // PUPIL DETECTION
    console.log("Detecting left pupil...");
    let leftPupil = detectPupilHoughFast(leftROI);
    console.log("Left pupil result:", leftPupil);
    
    console.log("Detecting right pupil...");
    let rightPupil = detectPupilHoughFast(rightROI);
    console.log("Right pupil result:", rightPupil);

    if (!leftPupil || !rightPupil) {
        showResult("❌ Pupil tidak ditemukan.");
        cleanup(src, gray, faceGray, leftROI, rightROI);
        return;
    }

    // Normalisasi SAMA seperti Python
    // left_normalized = left_pupil_x / left_eye_w
    let leftNorm = leftPupil.x / leftEyeRect.width;
    let rightNorm = rightPupil.x / rightEyeRect.width;
    let dx = leftNorm - rightNorm;
    let avgPos = (leftNorm + rightNorm) / 2;

    console.log("Left Normalized X:", leftNorm);
    console.log("Right Normalized X:", rightNorm);
    console.log("Difference (dx):", dx);
    console.log("Average Position:", avgPos);

    // Draw visualizations
    drawEyeVisualization(leftROI, leftPupil, "leftEyeCanvas", "Left Eye");
    drawEyeVisualization(rightROI, rightPupil, "rightEyeCanvas", "Right Eye");

    let leftNorm = leftPupil.x / leftROI.cols;
    let rightNorm = rightPupil.x / rightROI.cols;
    let dx = leftNorm - rightNorm;

    let html = `
        <div class="result-item"><b>Left Pupil:</b> (x: ${leftPupil.x}, y: ${leftPupil.y}, r: ${leftPupil.r})</div>
        <div class="result-item"><b>Right Pupil:</b> (x: ${rightPupil.x}, y: ${rightPupil.y}, r: ${rightPupil.r})</div>
        <div class="result-item"><b>Left Normalized X:</b> ${leftNorm.toFixed(3)}</div>
        <div class="result-item"><b>Right Normalized X:</b> ${rightNorm.toFixed(3)}</div>
        <div class="result-item"><b>Difference (dx):</b> ${dx.toFixed(3)}</div>
        <div class="result-item"><b>Average Position:</b> ${((leftNorm + rightNorm)/2).toFixed(3)}</div>
        <div class="result-item" style="margin-top: 15px; font-style: italic;">
            Nilai dx untuk gambar ini: ${dx.toFixed(6)}
        </div>
    `;

    showResult(html);

    cleanup(src, gray, faceGray, leftROI, rightROI);
}

// ===== Draw Eye Visualization =====
function drawEyeVisualization(eyeMat, pupil, canvasId, eyeTitle) {
    // Convert to RGB for visualization
    let output = new cv.Mat();
    cv.cvtColor(eyeMat, output, cv.COLOR_GRAY2RGB);

    // Draw green circle around pupil (SAMA seperti Python: linewidth=2)
    let center = new cv.Point(pupil.x, pupil.y);
    cv.circle(output, center, pupil.r, [0, 255, 0, 255], 2);
    
    // Draw red dot at center (SAMA seperti Python: size=3)
    cv.circle(output, center, 2, [255, 0, 0, 255], 3);

    // Display on canvas
    let canvas = document.getElementById(canvasId);
    cv.imshow(canvas, output);

    console.log(`${eyeTitle} - Pupil at (${pupil.x}, ${pupil.y}) with radius ${pupil.r}`);

    output.delete();
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

// ANALYZE BUTTON (setelah semua fungsi didefinisikan)
analyzeBtn.addEventListener("click", () => {
    // Check 1: OpenCV ready
    if (!cvReady) {
        alert("⏳ OpenCV belum siap. Tunggu beberapa detik lalu coba lagi.");
        return;
    }

    // Check 2: Cascades loaded
    if (!cascadesReady || !faceCascade || !eyeCascade) {
        alert("⏳ Haar Cascades belum siap. Tunggu beberapa detik lalu coba lagi.");
        return;
    }

    loadingDiv.style.display = "block";
    resultBox.style.display = "none";

    setTimeout(runAnalysis, 200);
});

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => m && m.delete());
}

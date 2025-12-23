let faceCascade, eyeCascade;
let cvReady = false;
let cascadesReady = false;

// OpenCV initialization
function onOpenCvReady() {
    console.log("OpenCV loading...");
    
    let checkInterval = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat && cv.CascadeClassifier) {
            clearInterval(checkInterval);
            console.log("✅ OpenCV is ready!");
            cvReady = true;
            loadCascadeFilesSequential();
        }
    }, 100);
}

function loadCascadeFilesSequential() {
    console.log("Loading Haar Cascades...");
    
    loadCascade("haarcascade_frontalface_default.xml", (classifier) => {
        faceCascade = classifier;
        console.log("✅ Face cascade loaded");
        
        loadCascade("haarcascade_eye.xml", (classifier) => {
            eyeCascade = classifier;
            console.log("✅ Eye cascade loaded");
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

// DOM Elements
const fileInput = document.getElementById("fileInput");
const filePreview = document.getElementById("filePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingDiv = document.getElementById("loading");
const resultBox = document.getElementById("results");
const resultContent = document.getElementById("resultContent");

fileInput.addEventListener("change", () => {
    let file = fileInput.files[0];
    if (!file) return;
    filePreview.src = URL.createObjectURL(file);
    filePreview.style.display = "block";
    analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener("click", () => {
    if (!cvReady || !cascadesReady) {
        alert("OpenCV atau Haar Cascades belum siap!");
        return;
    }
    
    loadingDiv.style.display = "block";
    resultBox.style.display = "none";
    
    setTimeout(() => {
        runAnalysis();
    }, 100);
});

// ===== FUNGSI UTAMA DARI PYTHON (DIUBAH KE JAVASCRIPT) =====

// ========================================
// IMPROVED FACE DETECTION (AUTO PARAMETER)
// ========================================
function improved_face_detection(gray) {
    console.log("\n🔍 Mencari wajah dengan berbagai kombinasi parameter...");
    console.log("📊 Input gray size:", gray.cols, "x", gray.rows);

    // Parameter yang akan dicoba (SAMA PERSIS dengan Python)
    const scaleFactors = [1.05, 1.1, 1.15, 1.2];
    const minNeighbors_list = [3, 4, 5, 6];
    const minSizes = [
        new cv.Size(50, 50),
        new cv.Size(60, 60),
        new cv.Size(80, 80),
        new cv.Size(100, 100)
    ];

    let best_face = null;
    let best_score = -1;
    let best_params = null;

    for (let sf of scaleFactors) {
        for (let mn of minNeighbors_list) {
            for (let ms of minSizes) {
                let faces = new cv.RectVector();
                
                try {
                    faceCascade.detectMultiScale(
                        gray, faces, sf, mn, 0, ms, new cv.Size(0, 0)
                    );

                    if (faces.size() > 0) {
                        // Ambil wajah dengan area terbesar
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
                        facesList.sort((a, b) => b.area - a.area);
                        
                        let fx = facesList[0].x, fy = facesList[0].y;
                        let fw = facesList[0].width, fh = facesList[0].height;

                        // Hitung score berdasarkan area dan rasio aspek (SAMA dengan Python)
                        let area = fw * fh;
                        let aspect_ratio = fw / fh;
                        let aspect_penalty = Math.abs(1.0 - aspect_ratio);
                        let score = area * (1 - aspect_penalty * 0.5);

                        if (score > best_score) {
                            best_score = score;
                            best_face = [Math.round(fx), Math.round(fy), Math.round(fw), Math.round(fh)];
                            best_params = [sf, mn, [ms.width, ms.height]];
                        }
                    }
                } catch(e) {
                    console.warn("Error with params sf=" + sf + ", mn=" + mn + ":", e.message);
                    continue;
                }
                
                faces.delete();
            }
        }
    }

    if (best_face === null) {
        console.log("❌ Tidak ada wajah terdeteksi dengan semua kombinasi parameter");
        return [null, null, null];
    }

    console.log("✅ Wajah terdeteksi!");
    console.log("   Parameter terbaik: scaleFactor=" + best_params[0] + 
                ", minNeighbors=" + best_params[1] + 
                ", minSize=" + best_params[2]);

    let [fx, fy, fw, fh] = best_face;
    console.log(`📊 Info Wajah: x=${fx}, y=${fy}, width=${fw}, height=${fh}`);

    // Crop wajah (setengah atas untuk area mata) - SAMA dengan Python
    let face_roi = new cv.Rect(fx, fy, fw, Math.floor(fh/2));
    let face_gray = new cv.Mat();
    face_gray = gray.roi(face_roi);
    
    console.log("📊 Face cropped size:", face_gray.cols, "x", face_gray.rows);

    return [best_face, face_gray, best_params];
}

// ========================================
// IMPROVED EYE DETECTION (AUTO PARAMETER)
// ========================================
function improved_eye_detection(gray) {
    console.log("\n🔍 Mencari mata dengan berbagai kombinasi parameter...");
    console.log("📊 Input gray size:", gray.cols, "x", gray.rows);

    // Parameter yang lebih sensitif untuk gambar close-up (SAMA PERSIS dengan Python)
    const scaleFactors = [1.01, 1.02, 1.05, 1.1];
    const minNeighbors_list = [1, 2, 3];
    const minSizes = [
        new cv.Size(10, 10),
        new cv.Size(15, 15),
        new cv.Size(20, 20),
        new cv.Size(25, 25)
    ];

    let best_eyes = null;
    let best_score = -1;
    let best_params = null;

    for (let sf of scaleFactors) {
        for (let mn of minNeighbors_list) {
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                
                try {
                    eyeCascade.detectMultiScale(
                        gray, eyes, sf, mn, 0, ms, new cv.Size(0, 0)
                    );

                    // Filter: hanya ambil 2 mata dengan area terbesar
                    if (eyes.size() >= 2) {
                        let areas = [];
                        for (let i = 0; i < eyes.size(); i++) {
                            let eye = eyes.get(i);
                            areas.push([eye.x, eye.y, eye.width, eye.height, eye.width * eye.height]);
                        }
                        
                        areas.sort((a, b) => b[4] - a[4]);
                        let areas_sorted = areas.slice(0, 2);

                        // Pastikan kedua mata tidak tumpang tindih
                        let eye1 = areas_sorted[0].slice(0, 4);
                        let eye2 = areas_sorted[1].slice(0, 4);
                        
                        let x1 = eye1[0], y1 = eye1[1], w1 = eye1[2], h1 = eye1[3];
                        let x2 = eye2[0], y2 = eye2[1], w2 = eye2[2], h2 = eye2[3];

                        // Hitung overlap (SAMA dengan Python)
                        let dx = Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2);
                        let dy = Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2);

                        if (dx <= 0 || dy <= 0) {  // Tidak overlap
                            let score = areas_sorted[0][4] + areas_sorted[1][4];
                            if (score > best_score) {
                                best_score = score;
                                best_eyes = [
                                    [Math.round(x1), Math.round(y1), Math.round(w1), Math.round(h1)],
                                    [Math.round(x2), Math.round(y2), Math.round(w2), Math.round(h2)]
                                ];
                                best_params = [sf, mn, [ms.width, ms.height]];
                            }
                        }
                    }
                } catch(e) {
                    console.warn("Error with params sf=" + sf + ", mn=" + mn + ":", e.message);
                    continue;
                }
                
                eyes.delete();
            }
        }
    }

    if (best_eyes === null || best_eyes.length < 2) {
        console.log("❌ Tidak dapat mendeteksi 2 mata");
        return [[], null];
    }

    console.log("✅ Kedua mata terdeteksi!");
    console.log("   Parameter terbaik: scaleFactor=" + best_params[0] + 
                ", minNeighbors=" + best_params[1] + 
                ", minSize=" + best_params[2]);
    console.log("📊 Mata kiri:", best_eyes[0]);
    console.log("📊 Mata kanan:", best_eyes[1]);

    return [best_eyes, best_params];
}

// ========================================
// IMPROVED PUPIL DETECTION (HOUGH TRANSFORM)
// ========================================
function detect_pupil_improved(eye_region, eye_name) {
    console.log(`\n🔍 Detecting pupil for ${eye_name}...`);
    console.log("📊 Eye region size:", eye_region.cols, "x", eye_region.rows);
    
    const strategies = {
        'threshold_simple': (eye) => {
            let processed = new cv.Mat();
            cv.threshold(eye, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            return processed;
        },
        'adaptive_threshold': (eye) => {
            let processed = new cv.Mat();
            cv.adaptiveThreshold(eye, processed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 11, 2);
            return processed;
        },
        'hist_eq': (eye) => {
            let equalized = new cv.Mat();
            cv.equalizeHist(eye, equalized);
            let processed = new cv.Mat();
            cv.threshold(equalized, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            equalized.delete();
            return processed;
        },
        'blur_threshold': (eye) => {
            let blurred = new cv.Mat();
            cv.GaussianBlur(eye, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
            let processed = new cv.Mat();
            cv.threshold(blurred, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            blurred.delete();
            return processed;
        },
        'morph_open': (eye) => {
            let processed = new cv.Mat();
            cv.threshold(eye, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
            cv.morphologyEx(processed, processed, cv.MORPH_OPEN, kernel);
            kernel.delete();
            return processed;
        }
    };

    let best_circle = null;
    let best_score = -1;
    let best_strategy = '';

    for (let [strategy_name, strategy_func] of Object.entries(strategies)) {
        try {
            let processed = strategy_func(eye_region);
            
            let circles = new cv.Mat();
            cv.HoughCircles(
                processed,
                circles,
                cv.HOUGH_GRADIENT,
                1,                                    // dp
                eye_region.rows / 8,                  // minDist
                100,                                  // param1
                15,                                   // param2
                Math.floor(eye_region.cols * 0.15),   // minRadius
                Math.floor(eye_region.cols * 0.4)     // maxRadius
            );

            console.log(`   Strategy ${strategy_name}: ${circles.cols} circle(s) found`);

            if (circles.cols > 0) {
                for (let i = 0; i < circles.cols; i++) {
                    let x = circles.data32F[i * 3];
                    let y = circles.data32F[i * 3 + 1];
                    let r = circles.data32F[i * 3 + 2];

                    // Hitung intensitas rata-rata dalam circle
                    let mask = cv.Mat.zeros(eye_region.rows, eye_region.cols, cv.CV_8UC1);
                    cv.circle(mask, new cv.Point(x, y), r, [255, 255, 255, 255], -1);
                    let mean_intensity = cv.mean(eye_region, mask)[0];
                    mask.delete();

                    console.log(`      Circle ${i}: x=${x.toFixed(1)}, y=${y.toFixed(1)}, r=${r.toFixed(1)}, intensity=${mean_intensity.toFixed(1)}`);

                    // Score: prioritaskan intensitas rendah (gelap)
                    let intensity_score = Math.max(0, 100 - mean_intensity) / 100;
                    let radius_score = r / eye_region.cols;
                    let center_score = 1.0 - (Math.abs(x - eye_region.cols/2) / (eye_region.cols/2));
                    
                    let score = intensity_score * 0.6 + radius_score * 0.2 + center_score * 0.2;

                    console.log(`      Score: ${score.toFixed(3)} (intensity: ${intensity_score.toFixed(3)}, radius: ${radius_score.toFixed(3)}, center: ${center_score.toFixed(3)})`);

                    // Filter: intensitas harus di bawah threshold
                    if (mean_intensity < 100 && score > best_score) {
                        best_score = score;
                        best_circle = [Math.round(x), Math.round(y), Math.round(r)];
                        best_strategy = strategy_name;
                    }
                }
            }

            circles.delete();
            processed.delete();
        } catch(e) {
            console.warn(`   Strategy ${strategy_name} failed:`, e.message);
            continue;
        }
    }

    if (best_circle) {
        let [x, y, r] = best_circle;
        let mask = cv.Mat.zeros(eye_region.rows, eye_region.cols, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(x, y), r, [255, 255, 255, 255], -1);
        let final_intensity = cv.mean(eye_region, mask)[0];
        mask.delete();

        console.log(`\n✅ TERPILIH [${best_strategy}]:`);
        console.log(`   Position: (${x}, ${y})`);
        console.log(`   Radius: ${r} px`);
        console.log(`   Intensity: ${final_intensity.toFixed(1)}`);
        console.log(`   Score: ${best_score.toFixed(3)}`);

        if (final_intensity > 80) {
            console.log(`   ⚠️  WARNING: Intensity tinggi! Mungkin bukan pupil.`);
        }
    } else {
        console.log(`\n❌ Tidak ada pupil terdeteksi`);
    }

    return best_circle;
}

function detect_pupil_with_fallback(eye_region, eye_name) {
    let circle = detect_pupil_improved(eye_region, eye_name);

    if (circle) {
        let mask = cv.Mat.zeros(eye_region.rows, eye_region.cols, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(circle[0], circle[1]), circle[2], [255, 255, 255, 255], -1);
        let intensity = cv.mean(eye_region, mask)[0];
        mask.delete();

        let confidence;
        if (intensity < 60) {
            confidence = "high";
        } else if (intensity < 80) {
            confidence = "medium";
        } else {
            confidence = "low";
        }

        return { circle, confidence };
    }

    return { circle: null, confidence: "failed" };
}

// ========================================
// CALCULATE PUPIL POSITION DIFFERENCE
// ========================================
function calculate_pupil_position_difference(pupils, eyes) {
    if (!pupils[0].circle || !pupils[1].circle) {
        console.log("Pupil tidak terdeteksi pada salah satu atau kedua mata");
        return null;
    }

    let left_pupil_x = pupils[0].circle[0];
    let right_pupil_x = pupils[1].circle[0];

    let left_eye_w = eyes[0][2];
    let right_eye_w = eyes[1][2];

    // Normalize pupil positions (SAMA dengan Python)
    let left_normalized = left_pupil_x / left_eye_w;
    let right_normalized = right_pupil_x / right_eye_w;

    let dx = left_normalized - right_normalized;

    return {
        dx: dx,
        left_normalized: left_normalized,
        right_normalized: right_normalized
    };
}

// ========================================
// MAIN ANALYSIS FUNCTION
// ========================================
function runAnalysis() {
    console.log("\n" + "=".repeat(60));
    console.log("STARTING STRABISMUS DETECTION");
    console.log("=".repeat(60));
    
    // Load image
    let img = filePreview;
    let src = cv.imread(img);
    
    console.log("\n📊 ORIGINAL IMAGE:");
    console.log("   Size:", src.cols, "x", src.rows);
    console.log("   Channels:", src.channels());
    console.log("   Type:", src.type());
    
    // **FIX 1: Convert RGBA to BGR (seperti Python)**
    // Python default baca BGR, JavaScript/Browser default RGBA
    let src_bgr = new cv.Mat();
    cv.cvtColor(src, src_bgr, cv.COLOR_RGBA2BGR);
    console.log("\n✅ Converted RGBA to BGR");
    console.log("   BGR Size:", src_bgr.cols, "x", src_bgr.rows);
    console.log("   BGR Channels:", src_bgr.channels());
    
    // Convert BGR to grayscale (sama seperti Python)
    let gray = new cv.Mat();
    cv.cvtColor(src_bgr, gray, cv.COLOR_BGR2GRAY);
    
    console.log("\n✅ Converted BGR to GRAY");
    console.log("   Gray Size:", gray.cols, "x", gray.rows);
    console.log("   Gray Channels:", gray.channels());

    // Step 1: Improved Face Detection
    console.log("\n" + "=".repeat(60));
    console.log("STEP 1: IMPROVED FACE DETECTION");
    console.log("=".repeat(60));
    
    let [face_coords, face_gray, face_params] = improved_face_detection(gray);

    if (face_coords === null) {
        showResult("❌ Gagal mendeteksi wajah. Coba foto lain dengan pencahayaan lebih baik.");
        cleanup(src, src_bgr, gray);
        return;
    }

    let [fx, fy, fw, fh] = face_coords;

    // Step 2: Improved Eye Detection
    console.log("\n" + "=".repeat(60));
    console.log("STEP 2: IMPROVED EYE DETECTION");
    console.log("=".repeat(60));
    
    let [eyes_in_face, eye_params] = improved_eye_detection(face_gray);

    if (eyes_in_face.length < 2) {
        console.log("❌ Mata tidak cukup terdeteksi");
        showResult("❌ Tidak dapat mendeteksi kedua mata. Pastikan mata terlihat jelas.");
        cleanup(src, src_bgr, gray, face_gray);
        return;
    }

    // Sort mata dari kiri ke kanan
    let eyes = eyes_in_face.sort((a, b) => a[0] - b[0]);
    console.log("📊 Final eyes (sorted):", eyes);

    // Step 3: Extract Eye ROIs
    console.log("\n" + "=".repeat(60));
    console.log("STEP 3: EXTRACT EYE REGIONS");
    console.log("=".repeat(60));
    
    try {
        // Left eye
        let left_eye_rect = new cv.Rect(eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]);
        let left_eye = new cv.Mat();
        left_eye = face_gray.roi(left_eye_rect);
        console.log("📊 Left eye extracted, size:", left_eye.cols, "x", left_eye.rows);

        // Right eye
        let right_eye_rect = new cv.Rect(eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]);
        let right_eye = new cv.Mat();
        right_eye = face_gray.roi(right_eye_rect);
        console.log("📊 Right eye extracted, size:", right_eye.cols, "x", right_eye.rows);

        // Step 4: Improved Pupil Detection
        console.log("\n" + "=".repeat(60));
        console.log("STEP 4: IMPROVED PUPIL DETECTION");
        console.log("=".repeat(60));
        
        let left_result = detect_pupil_with_fallback(left_eye, "Left Eye");
        let right_result = detect_pupil_with_fallback(right_eye, "Right Eye");

        console.log(`\n📊 PUPIL DETECTION RESULTS:`);
        console.log(`   Left confidence: ${left_result.confidence}`);
        console.log(`   Right confidence: ${right_result.confidence}`);

        let pupils = [left_result, right_result];

        // Step 5: Calculate Pupil Position Difference
        console.log("\n" + "=".repeat(60));
        console.log("STEP 5: PUPIL POSITION DIFFERENCE ANALYSIS");
        console.log("=".repeat(60));
        
        let result = calculate_pupil_position_difference(pupils, eyes);

        if (result !== null) {
            let { dx, left_normalized, right_normalized } = result;

            console.log(`\n📊 FINAL RESULTS:`);
            console.log(`   Left Eye Normalized X:  ${left_normalized.toFixed(3)}`);
            console.log(`   Right Eye Normalized X: ${right_normalized.toFixed(3)}`);
            console.log(`   Difference (dx):        ${dx.toFixed(3)}`);
            console.log(`   Average Position:       ${((left_normalized + right_normalized) / 2).toFixed(3)}`);
            console.log(`   Nilai dx: ${dx.toFixed(6)}`);

            // Visualisasi
            visualize_eye_result(left_eye, left_result, "leftEyeCanvas", "Left Eye", left_result.confidence);
            visualize_eye_result(right_eye, right_result, "rightEyeCanvas", "Right Eye", right_result.confidence);

            // Interpretasi (SAMA dengan Python)
            let interpretation = "";
            if (dx < -0.05) {
                interpretation = "→ Kemungkinan ESOTROPIA (mata menyerong ke dalam)";
            } else if (dx > 0.05) {
                interpretation = "→ Kemungkinan EXOTROPIA (mata menyerong ke luar)";
            } else {
                interpretation = "→ Kemungkinan NORMAL";
            }

            console.log(`\n💡 INTERPRETASI: ${interpretation}`);

            let htmlOutput = `
                <div class="result-item"><b>Left Pupil:</b> (x: ${left_result.circle ? left_result.circle[0] : 'N/A'}, 
                y: ${left_result.circle ? left_result.circle[1] : 'N/A'}, 
                r: ${left_result.circle ? left_result.circle[2] : 'N/A'}) 
                <span class="confidence-${left_result.confidence}">[${left_result.confidence}]</span></div>
                
                <div class="result-item"><b>Right Pupil:</b> (x: ${right_result.circle ? right_result.circle[0] : 'N/A'}, 
                y: ${right_result.circle ? right_result.circle[1] : 'N/A'}, 
                r: ${right_result.circle ? right_result.circle[2] : 'N/A'}) 
                <span class="confidence-${right_result.confidence}">[${right_result.confidence}]</span></div>
                
                <div class="result-item"><b>Left Eye Normalized X:</b> ${left_normalized.toFixed(3)}</div>
                <div class="result-item"><b>Right Eye Normalized X:</b> ${right_normalized.toFixed(3)}</div>
                <div class="result-item"><b>Difference (dx):</b> ${dx.toFixed(3)}</div>
                <div class="result-item"><b>Average Position:</b> ${((left_normalized + right_normalized) / 2).toFixed(3)}</div>
                
                <div class="result-item" style="margin-top: 15px; padding: 15px; background: #e8f4fd; border-left: 4px solid #007bff;">
                    <b>💡 Interpretasi:</b><br>
                    ${interpretation}
                </div>
                
                <div class="result-item" style="margin-top: 15px; font-style: italic; color: #007bff;">
                    <b>Nilai dx untuk gambar ini:</b> ${dx.toFixed(6)}
                </div>
            `;
            showResult(htmlOutput);
        } else {
            console.log("❌ Tidak dapat menghitung perbedaan posisi pupil");
            showResult("❌ Tidak dapat menghitung perbedaan posisi pupil. Pupil tidak terdeteksi dengan baik.");
        }

        cleanup(src, src_bgr, gray, face_gray, left_eye, right_eye);
        
        console.log("\n" + "=".repeat(60));
        console.log("ANALYSIS COMPLETE");
        console.log("=".repeat(60));
        
    } catch(error) {
        console.error("\n❌ ERROR during analysis:", error);
        console.error("Error stack:", error.stack);
        showResult(`❌ Error: ${error.message}`);
        cleanup(src, src_bgr, gray, face_gray);
    }
}

function visualize_eye_result(eye_region, result, canvasId, eye_name, confidence) {
    let output = new cv.Mat();
    cv.cvtColor(eye_region, output, cv.COLOR_GRAY2RGB);

    if (result.circle !== null) {
        let [x, y, r] = result.circle;

        // Warna berdasarkan confidence (SAMA dengan Python)
        let color;
        if (confidence === "high") {
            color = [0, 255, 0, 255];  // Hijau
        } else if (confidence === "medium") {
            color = [255, 255, 0, 255];  // Kuning
        } else if (confidence === "low") {
            color = [255, 165, 0, 255];  // Orange
        } else {
            color = [0, 255, 0, 255];  // Default hijau
        }

        cv.circle(output, new cv.Point(x, y), r, color, 2);
        cv.circle(output, new cv.Point(x, y), 2, [255, 0, 0, 255], 3);

        // Add confidence label
        let label = `Conf: ${confidence}`;
        cv.putText(output, label, new cv.Point(5, 15), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255, 255], 1);
    }

    let canvas = document.getElementById(canvasId);
    cv.imshow(canvas, output);
    output.delete();
    
    // Update confidence badge
    document.getElementById(canvasId.replace('Canvas', 'Confidence')).innerHTML = 
        `<span class="confidence-${confidence}">[${confidence}]</span>`;
}

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => {
        if (m && m.delete) {
            try {
                m.delete();
            } catch(e) {
                console.warn("Error deleting mat:", e.message);
            }
        }
    });
}

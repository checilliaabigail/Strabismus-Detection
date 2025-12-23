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

// ===== FUNGSI UTAMA DARI PYTHON (DIUBAH KE JAVASCRIPT) =====

// ========================================
// IMPROVED FACE DETECTION (AUTO PARAMETER)
// ========================================
function improved_face_detection(gray) {
    console.log("🔍 Mencari wajah dengan berbagai kombinasi parameter...");

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
                        gray, faces, sf, mn, 0, ms
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

    // Crop wajah (setengah atas untuk area mata) - SAMA dengan Python
    let face_roi = new cv.Rect(fx, fy, fw, Math.floor(fh/2));
    let face_gray = gray.roi(face_roi);

    return [best_face, face_gray, best_params];
}

// ========================================
// IMPROVED EYE DETECTION (AUTO PARAMETER)
// ========================================
function improved_eye_detection(gray) {
    console.log("🔍 Mencari mata dengan berbagai kombinasi parameter...");

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
                        gray, eyes, sf, mn, 0, ms
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
                                    [Math.round(eye1[0]), Math.round(eye1[1]), Math.round(eye1[2]), Math.round(eye1[3])],
                                    [Math.round(eye2[0]), Math.round(eye2[1]), Math.round(eye2[2]), Math.round(eye2[3])]
                                ];
                                best_params = [sf, mn, [ms.width, ms.height]];
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

    if (best_eyes === null || best_eyes.length < 2) {
        console.log("❌ Tidak dapat mendeteksi 2 mata");
        return [[], null];
    }

    console.log("✅ Kedua mata terdeteksi!");
    console.log("   Parameter terbaik: scaleFactor=" + best_params[0] + 
                ", minNeighbors=" + best_params[1] + 
                ", minSize=" + best_params[2]);

    return [best_eyes, best_params];
}

// ========================================
// IMPROVED PUPIL DETECTION (HOUGH TRANSFORM)
// ========================================
function detect_pupil_improved(eye_region, eye_name = "Eye", debug = false) {
    if (!eye_region || eye_region.rows === 0 || eye_region.cols === 0) {
        console.log(`\n❌ ${eye_name}: Region mata kosong`);
        return null;
    }

    let h = eye_region.rows, w = eye_region.cols;
    console.log(`\n${'='*60}`);
    console.log(`Deteksi Pupil: ${eye_name}`);
    console.log(`${'='*60}`);
    console.log(`Ukuran eye region: ${w} x ${h} px`);

    // ========================================
    // PREPROCESSING UNTUK ISOLASI PUPIL
    // ========================================

    // 1. Gaussian blur untuk noise reduction
    let blurred = new cv.Mat();
    cv.GaussianBlur(eye_region, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    // 2. CLAHE untuk enhance contrast
    let clahe = new cv.CLAHE();
    clahe.setClipLimit(2.0);
    clahe.setTilesGridSize(new cv.Size(8, 8));
    
    let enhanced = new cv.Mat();
    clahe.apply(blurred, enhanced);

    // 3. Binary threshold
    let binary = new cv.Mat();
    cv.threshold(enhanced, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    
    let binary_inv = new cv.Mat();
    cv.bitwise_not(binary, binary_inv);

    // 4. Morphological closing untuk fill small holes
    let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    let cleaned = new cv.Mat();
    cv.morphologyEx(binary_inv, cleaned, cv.MORPH_CLOSE, kernel);

    // Estimasi ukuran pupil (12-25% dari lebar eye region) - SAMA dengan Python
    let min_radius = Math.max(5, Math.floor(w * 0.12));
    let max_radius = Math.min(Math.floor(w * 0.25), 40);

    console.log(`Estimasi radius pupil: ${min_radius}-${max_radius} px`);

    let center_x = Math.floor(w / 2);
    let center_y = Math.floor(h / 2);

    // ========================================
    // DETEKSI DENGAN MULTIPLE STRATEGIES
    // ========================================

    let strategies = [
        { name: "Enhanced", img: enhanced },
        { name: "Binary", img: cleaned }
    ];

    let best_circle = null;
    let best_score = -1;
    let best_strategy = null;

    for (let strategy of strategies) {
        // Parameter combinations (SAMA dengan Python)
        let params_list = [
            { dp: 1.2, minDist: 15, param1: 50, param2: 30 },  // Konservatif
            { dp: 1.1, minDist: 15, param1: 50, param2: 25 },  // Balanced
            { dp: 1.1, minDist: 12, param1: 45, param2: 22 },  // Lebih sensitif
            { dp: 1.2, minDist: 15, param1: 40, param2: 28 },  // Alternative
        ];

        for (let params of params_list) {
            try {
                let circles = new cv.Mat();
                cv.HoughCircles(
                    strategy.img,
                    circles,
                    cv.HOUGH_GRADIENT,
                    params.dp,
                    params.minDist,
                    params.param1,
                    params.param2,
                    min_radius,
                    max_radius
                );

                if (circles.cols > 0) {
                    for (let i = 0; i < circles.cols; i++) {
                        let x = Math.round(circles.data32F[i * 3]);
                        let y = Math.round(circles.data32F[i * 3 + 1]);
                        let r = Math.round(circles.data32F[i * 3 + 2]);

                        // Validasi 1: Dalam batas
                        if (x - r < 0 || x + r > w || y - r < 0 || y + r > h) {
                            continue;
                        }

                        // Validasi 2: Tidak terlalu dekat dengan tepi
                        let edge_margin = 5;
                        if (x < edge_margin || x > w - edge_margin || 
                            y < edge_margin || y > h - edge_margin) {
                            continue;
                        }

                        // ========================================
                        // SCORING DENGAN VALIDASI KETAT (SAMA dengan Python)
                        // ========================================

                        // 1. Kedekatan dengan center (40%)
                        let dist = Math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2);
                        let center_score = 1.0 / (1.0 + dist / (w * 0.25));

                        // 2. Intensitas - PUPIL HARUS GELAP! (40%)
                        let mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);
                        cv.circle(mask, new cv.Point(x, y), r, new cv.Scalar(255), -1);
                        
                        let mean_intensity = cv.mean(eye_region, mask)[0];
                        
                        // ⚠️ KRITERIA KETAT: Pupil harus < 80 intensity (SAMA dengan Python)
                        if (mean_intensity > 100) {
                            mask.delete();
                            continue;
                        }

                        // Semakin gelap semakin baik
                        let intensity_score = Math.max(0, 1.0 - (mean_intensity / 80.0));

                        // 3. Ukuran radius (15%)
                        let optimal_r = (min_radius + max_radius) / 2;
                        let size_score = 1.0 - Math.abs(r - optimal_r) / max_radius;

                        // 4. Variance intensitas (5%) - pupil homogen
                        // Note: OpenCV.js tidak punya std dev langsung, kita gunakan approximation
                        let variance_score = 1.0 / (1.0 + mean_intensity / 50.0);

                        // Total score (WEIGHT SAMA dengan Python)
                        let score = (
                            center_score * 0.40 +
                            intensity_score * 0.40 +
                            size_score * 0.15 +
                            variance_score * 0.05
                        );

                        console.log(`  [${strategy.name}] x=${x}, y=${y}, r=${r}, intensity=${mean_intensity.toFixed(1)}, score=${score.toFixed(3)}`);

                        if (score > best_score) {
                            best_score = score;
                            best_circle = [x, y, r];
                            best_strategy = strategy.name;
                        }

                        mask.delete();
                    }
                }
                circles.delete();
            } catch(e) {
                continue;
            }
        }
    }

    // Cleanup mats
    blurred.delete();
    enhanced.delete();
    binary.delete();
    binary_inv.delete();
    cleaned.delete();
    clahe.delete();

    // ========================================
    // HASIL
    // ========================================

    if (best_circle) {
        let [x, y, r] = best_circle;
        
        // Hitung final intensity untuk validasi
        let mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(x, y), r, new cv.Scalar(255), -1);
        let final_intensity = cv.mean(eye_region, mask)[0];
        mask.delete();

        console.log(`\n✅ TERPILIH [${best_strategy}]:`);
        console.log(`   Position: (${x}, ${y})`);
        console.log(`   Radius: ${r} px`);
        console.log(`   Intensity: ${final_intensity.toFixed(1)}`);
        console.log(`   Score: ${best_score.toFixed(3)}`);

        // Warning jika intensity masih tinggi (SAMA dengan Python)
        if (final_intensity > 80) {
            console.log(`   ⚠️  WARNING: Intensity tinggi! Mungkin bukan pupil.`);
        }
    } else {
        console.log(`\n❌ Tidak ada pupil terdeteksi`);
    }

    return best_circle;
}

function detect_pupil_with_fallback(eye_region, eye_name = "Eye") {
    // Try improved detection first
    let circle = detect_pupil_improved(eye_region, eye_name, false);

    if (circle) {
        // Hitung confidence berdasarkan intensity (SAMA dengan Python)
        let [x, y, r] = circle;
        let h = eye_region.rows, w = eye_region.cols;
        
        let mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(x, y), r, new cv.Scalar(255), -1);
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
    if (pupils[0].circle === null || pupils[1].circle === null) {
        console.log("Pupil tidak terdeteksi pada salah satu atau kedua mata");
        return null;
    }

    // Extract pupil x positions and eye widths
    let left_pupil_x = pupils[0].circle[0];
    let right_pupil_x = pupils[1].circle[0];

    let left_eye_w = eyes[0][2];
    let right_eye_w = eyes[1][2];

    // Normalize pupil positions within their respective eye bounding boxes
    let left_normalized = left_pupil_x / left_eye_w;
    let right_normalized = right_pupil_x / right_eye_w;

    // Hitung perbedaan posisi antara kedua pupil
    let dx = left_normalized - right_normalized;

    return { dx, left_normalized, right_normalized };
}

// ========================================
// MAIN ANALYSIS FLOW
// ========================================
analyzeBtn.addEventListener("click", () => {
    if (!cvReady || !cascadesReady || !faceCascade || !eyeCascade) {
        alert("⏳ Tunggu OpenCV dan cascades siap.");
        return;
    }

    loadingDiv.style.display = "block";
    resultBox.style.display = "none";

    setTimeout(runAnalysis, 200);
});

function runAnalysis() {
    // Load image
    let img = filePreview;
    let src = cv.imread(img);
    
    console.log("Image loaded - Size:", src.cols, "x", src.rows, "Channels:", src.channels());
    
    // Convert to grayscale
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    
    console.log("Gray image - Size:", gray.cols, "x", gray.rows);

    // Step 1: Improved Face Detection
    console.log("\n" + "="*60);
    console.log("STEP 1: IMPROVED FACE DETECTION");
    console.log("="*60);
    
    let [face_coords, face_gray, face_params] = improved_face_detection(gray);

    if (face_coords === null) {
        showResult("❌ Gagal lanjut ke eye detection karena wajah tidak ditemukan.");
        cleanup(src, gray);
        return;
    }

    let [fx, fy, fw, fh] = face_coords;
    console.log(`✅ Wajah terdeteksi: x=${fx}, y=${fy}, w=${fw}, h=${fh}`);
    console.log(`📊 Parameter terbaik: scaleFactor=${face_params[0]}, minNeighbors=${face_params[1]}`);

    // Step 2: Improved Eye Detection
    console.log("\n" + "="*60);
    console.log("STEP 2: IMPROVED EYE DETECTION");
    console.log("="*60);
    
    let [eyes_in_face, eye_params] = improved_eye_detection(face_gray);

    if (eyes_in_face.length < 2) {
        console.log("Error: Not enough eyes detected");
        showResult("❌ Mata tidak ditemukan.");
        cleanup(src, gray, face_gray);
        return;
    }

    // Sort mata dari kiri ke kanan
    let eyes = eyes_in_face.sort((a, b) => a[0] - b[0]);
    console.log("Final eyes (sorted):", eyes);

    console.log(`📊 Info Mata:`);
    console.log(`   Mata kiri: [${eyes[0]}]`);
    console.log(`   Mata kanan: [${eyes[1]}]`);

    // Step 3: Extract Eye ROIs
    console.log("\n" + "="*60);
    console.log("STEP 3: EXTRACT EYE REGIONS");
    console.log("="*60);
    
    try {
        // Left eye
        let left_eye_rect = new cv.Rect(eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]);
        let left_eye = face_gray.roi(left_eye_rect);
        console.log("Left eye extracted, size:", left_eye.cols, "x", left_eye.rows);

        // Right eye
        let right_eye_rect = new cv.Rect(eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]);
        let right_eye = face_gray.roi(right_eye_rect);
        console.log("Right eye extracted, size:", right_eye.cols, "x", right_eye.rows);

        // Step 4: Improved Pupil Detection
        console.log("\n" + "="*60);
        console.log("STEP 4: IMPROVED PUPIL DETECTION");
        console.log("="*60);
        
        let left_result = detect_pupil_with_fallback(left_eye, "Left Eye");
        let right_result = detect_pupil_with_fallback(right_eye, "Right Eye");

        console.log(`\nConfidence: Left=${left_result.confidence}, Right=${right_result.confidence}`);

        let pupils = [left_result, right_result];

        // Step 5: Calculate Pupil Position Difference
        console.log("\n" + "="*60);
        console.log("STEP 5: PUPIL POSITION DIFFERENCE ANALYSIS");
        console.log("="*60);
        
        let result = calculate_pupil_position_difference(pupils, eyes);

        if (result !== null) {
            let { dx, left_normalized, right_normalized } = result;
            let dx_value = dx;

            console.log(`Left Eye Normalized X:  ${left_normalized.toFixed(3)}`);
            console.log(`Right Eye Normalized X: ${right_normalized.toFixed(3)}`);
            console.log(`Difference (dx):        ${dx.toFixed(3)}`);
            console.log(`Average Position:       ${((left_normalized + right_normalized) / 2).toFixed(3)}`);
            console.log(`\nNilai dx untuk gambar ini: ${dx_value.toFixed(6)}`);

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

            let htmlOutput = `
                <div class="result-item"><b>Left Pupil:</b> (x: ${left_result.circle ? left_result.circle[0] : 'N/A'}, 
                y: ${left_result.circle ? left_result.circle[1] : 'N/A'}, 
                r: ${left_result.circle ? left_result.circle[2] : 'N/A'}) 
                <span style="color: ${getConfidenceColor(left_result.confidence)}">[${left_result.confidence}]</span></div>
                
                <div class="result-item"><b>Right Pupil:</b> (x: ${right_result.circle ? right_result.circle[0] : 'N/A'}, 
                y: ${right_result.circle ? right_result.circle[1] : 'N/A'}, 
                r: ${right_result.circle ? right_result.circle[2] : 'N/A'}) 
                <span style="color: ${getConfidenceColor(right_result.confidence)}">[${right_result.confidence}]</span></div>
                
                <div class="result-item"><b>Left Eye Normalized X:</b> ${left_normalized.toFixed(3)}</div>
                <div class="result-item"><b>Right Eye Normalized X:</b> ${right_normalized.toFixed(3)}</div>
                <div class="result-item"><b>Difference (dx):</b> ${dx.toFixed(3)}</div>
                <div class="result-item"><b>Average Position:</b> ${((left_normalized + right_normalized) / 2).toFixed(3)}</div>
                
                <div class="result-item" style="margin-top: 15px; padding: 10px; background: #e8f4fd; border-left: 4px solid #007bff;">
                    <b>💡 Interpretasi:</b><br>
                    ${interpretation}<br>
                    <small style="color: #666;">(Gunakan threshold yang sudah Anda tentukan untuk klasifikasi final)</small>
                </div>
                
                <div class="result-item" style="margin-top: 15px; font-style: italic; color: #007bff;">
                    Nilai dx untuk gambar ini: ${dx_value.toFixed(6)}
                </div>
            `;
            showResult(htmlOutput);
        } else {
            console.log("Tidak dapat menghitung perbedaan posisi pupil");
            showResult("❌ Tidak dapat menghitung perbedaan posisi pupil");
        }

        cleanup(src, gray, face_gray, left_eye, right_eye);
        
    } catch(error) {
        console.error("ERROR during analysis:", error);
        console.error("Error stack:", error.stack);
        showResult(`❌ Error: ${error.message}`);
        cleanup(src, gray, face_gray);
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
}

function getConfidenceColor(confidence) {
    switch(confidence) {
        case "high": return "#28a745";
        case "medium": return "#ffc107";
        case "low": return "#fd7e14";
        case "failed": return "#dc3545";
        default: return "#6c757d";
    }
}

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => {
        if (m && m.delete) {
            m.delete();
        }
    });
}

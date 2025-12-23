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

// ========================================
// IMPROVED FACE DETECTION (FIXED)
// ========================================
function detect_face_improved(gray) {
    console.log("\n" + "="*60);
    console.log("FACE DETECTION");
    console.log("="*60);
    
    let faces = new cv.RectVector();
    let minSize = new cv.Size(80, 80);
    
    // Try multiple parameters like in Python
    const scaleFactors = [1.05, 1.1, 1.15, 1.2];
    const minNeighbors_list = [3, 4, 5, 6];
    
    let best_face = null;
    let best_score = -1;
    
    for (let sf of scaleFactors) {
        for (let mn of minNeighbors_list) {
            faces.delete();
            faces = new cv.RectVector();
            
            faceCascade.detectMultiScale(gray, faces, sf, mn, 0, minSize);
            
            if (faces.size() > 0) {
                // Get largest face
                let largest_face = null;
                let max_area = 0;
                
                for (let i = 0; i < faces.size(); i++) {
                    let face = faces.get(i);
                    let area = face.width * face.height;
                    
                    if (area > max_area) {
                        max_area = area;
                        largest_face = face;
                    }
                }
                
                // Calculate score like in Python
                let fx = largest_face.x, fy = largest_face.y;
                let fw = largest_face.width, fh = largest_face.height;
                
                let aspect_ratio = fw / fh;
                let aspect_penalty = Math.abs(1.0 - aspect_ratio);
                let score = max_area * (1 - aspect_penalty * 0.5);
                
                if (score > best_score) {
                    best_score = score;
                    best_face = [fx, fy, fw, fh];
                }
            }
        }
    }
    
    if (best_face === null) {
        console.log("❌ Tidak ada wajah terdeteksi");
        faces.delete();
        return [null, null];
    }
    
    let [fx, fy, fw, fh] = best_face;
    console.log(`✅ Wajah terdeteksi: ${fx}, ${fy}, ${fw}, ${fh}`);
    
    // Crop setengah atas wajah (upper half) seperti di Python
    let face_roi = new cv.Rect(fx, fy, fw, Math.floor(fh / 2));
    let face_gray = gray.roi(face_roi);
    
    faces.delete();
    return [best_face, face_gray];
}

// ========================================
// IMPROVED EYE DETECTION (FIXED)
// ========================================
function improved_eye_detection(face_gray) {
    console.log("\n" + "="*60);
    console.log("EYE DETECTION");
    console.log("="*60);
    
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
    
    for (let sf of scaleFactors) {
        for (let mn of minNeighbors_list) {
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                
                eyeCascade.detectMultiScale(face_gray, eyes, sf, mn, 0, ms);
                
                if (eyes.size() >= 2) {
                    // Convert to array with area
                    let eyesArray = [];
                    for (let i = 0; i < eyes.size(); i++) {
                        let eye = eyes.get(i);
                        eyesArray.push({
                            x: eye.x,
                            y: eye.y,
                            w: eye.width,
                            h: eye.height,
                            area: eye.width * eye.height
                        });
                    }
                    
                    // Sort by area descending
                    eyesArray.sort((a, b) => b.area - a.area);
                    let top2 = eyesArray.slice(0, 2);
                    
                    // Check if eyes don't overlap (like in Python)
                    let eye1 = top2[0];
                    let eye2 = top2[1];
                    
                    let dx = Math.min(eye1.x + eye1.w, eye2.x + eye2.w) - Math.max(eye1.x, eye2.x);
                    let dy = Math.min(eye1.y + eye1.h, eye2.y + eye2.h) - Math.max(eye1.y, eye2.y);
                    
                    if (dx <= 0 || dy <= 0) { // No overlap
                        let score = eye1.area + eye2.area;
                        
                        if (score > best_score) {
                            best_score = score;
                            best_eyes = [
                                [Math.round(eye1.x), Math.round(eye1.y), Math.round(eye1.w), Math.round(eye1.h)],
                                [Math.round(eye2.x), Math.round(eye2.y), Math.round(eye2.w), Math.round(eye2.h)]
                            ];
                        }
                    }
                }
                
                eyes.delete();
            }
        }
    }
    
    if (best_eyes === null) {
        console.log("❌ Mata tidak terdeteksi");
        return [[], null];
    }
    
    // Sort eyes from left to right
    best_eyes.sort((a, b) => a[0] - b[0]);
    console.log(`✅ Mata terdeteksi: Kiri=${best_eyes[0]}, Kanan=${best_eyes[1]}`);
    
    return [best_eyes, null];
}

// ========================================
// PUPIL DETECTION (SIMPLE HOUGH - FIXED & MATCH PYTHON)
// ========================================
function detect_pupil_hough_simple(eye_gray, eye_name = "Eye") {
    console.log(`\n${eye_name} PUPIL DETECTION`);
    
    if (!eye_gray || eye_gray.rows === 0 || eye_gray.cols === 0) {
        console.log("❌ Region mata kosong");
        return null;
    }
    
    let h = eye_gray.rows, w = eye_gray.cols;
    console.log(`Ukuran: ${w}x${h}`);
    
    // 1. Gaussian Blur - SAMA dengan Python: (7,7), 2
    let blurred = new cv.Mat();
    cv.GaussianBlur(eye_gray, blurred, new cv.Size(7, 7), 2, 2, cv.BORDER_DEFAULT);
    
    // 2. PARAMETER HOUGH SAMA DENGAN PYTHON
    let param_sets = [
        {dp: 1.1, minDist: 20, param1: 80, param2: 20, minR: 15, maxR: 40},
        {dp: 1.0, minDist: 20, param1: 60, param2: 15, minR: 10, maxR: 45},
        {dp: 0.9, minDist: 20, param1: 40, param2: 10, minR: 5, maxR: 50},
        {dp: 1.2, minDist: 20, param1: 100, param2: 25, minR: 20, maxR: 35},
        {dp: 1.0, minDist: 20, param1: 90, param2: 30, minR: 8, maxR: 30}
    ];
    
    let all_circles = []; // Simpan semua circle yang valid
    
    for (let params of param_sets) {
        try {
            let circles = new cv.Mat();
            cv.HoughCircles(
                blurred,
                circles,
                cv.HOUGH_GRADIENT,
                params.dp,
                params.minDist,
                params.param1,
                params.param2,
                params.minR,
                params.maxR
            );
            
            if (circles.cols > 0) {
                console.log(`  Ditemukan ${circles.cols} lingkaran dengan params: dp=${params.dp}`);
                
                // Process all detected circles
                for (let i = 0; i < circles.cols; i++) {
                    let x = Math.round(circles.data32F[i * 3]);
                    let y = Math.round(circles.data32F[i * 3 + 1]);
                    let r = Math.round(circles.data32F[i * 3 + 2]);
                    
                    // Validasi posisi (dalam bounds)
                    if (x - r > 0 && x + r < w && y - r > 0 && y + r < h) {
                        // Validasi radius (5-50 seperti di Python)
                        if (r >= 5 && r <= 50) {
                            all_circles.push([x, y, r]);
                        }
                    }
                }
            }
            circles.delete();
        } catch(e) {
            console.log(`Error dengan params:`, e);
            continue;
        }
    }
    
    // Cleanup blurred image
    blurred.delete();
    
    console.log(`Total circles found: ${all_circles.length}`);
    
    // Jika ada circles yang terdeteksi, pilih yang terbaik dengan scoring
    let best_circle = null;
    
    if (all_circles.length > 0) {
        let center_x = Math.floor(w / 2);
        let center_y = Math.floor(h / 2);
        
        let scored_circles = [];
        
        // SCORING SYSTEM SAMA DENGAN PYTHON
        for (let circle of all_circles) {
            let x = circle[0], y = circle[1], r = circle[2];
            
            // Hitung intensitas area circle
            let mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);
            cv.circle(mask, new cv.Point(x, y), r, new cv.Scalar(255), -1);
            let mean_intensity = cv.mean(eye_gray, mask)[0];
            mask.delete();
            
            // ========================================
            // SCORING SAMA PERSIS DENGAN PYTHON
            // ========================================
            
            // 1. Center score (40%) - semakin dekat center semakin baik
            let distance = Math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2);
            let center_score = 1.0 / (1.0 + distance / Math.max(w, h));
            
            // 2. Radius score (prefer 15-35 pixels)
            let radius_score;
            if (r >= 15 && r <= 35) {
                radius_score = 1.0;
            } else if (r >= 10 && r <= 40) {
                radius_score = 0.7;
            } else {
                radius_score = 0.3;
            }
            
            // 3. Intensity check - pupil harus gelap
            // Di Python ada validasi: if mean_intensity > 100: continue
            if (mean_intensity > 100) {
                // Skip jika terlalu terang
                continue;
            }
            
            // 4. Total score (70% center, 30% radius) - SAMA dengan Python
            let total_score = center_score * 0.7 + radius_score * 0.3;
            
            scored_circles.push({
                score: total_score,
                circle: [x, y, r],
                intensity: mean_intensity
            });
            
            console.log(`  Circle: (${x}, ${y}, r=${r}) intensity=${mean_intensity.toFixed(1)}, score=${total_score.toFixed(3)}`);
        }
        
        // Sort by score descending (terbaik pertama)
        scored_circles.sort((a, b) => b.score - a.score);
        
        if (scored_circles.length > 0) {
            best_circle = scored_circles[0].circle;
            console.log(`✅ Pupil terdeteksi: ${best_circle}, score=${scored_circles[0].score.toFixed(3)}, intensity=${scored_circles[0].intensity.toFixed(1)}`);
        }
    }
    
    if (!best_circle) {
        console.log("❌ Tidak ada pupil terdeteksi");
        return null;
    }
    
    return best_circle;
}

// ========================================
// CALCULATE PUPIL POSITION DIFFERENCE
// ========================================
function calculate_pupil_position_difference(pupils, eyes) {
    console.log("\n" + "="*60);
    console.log("CALCULATE PUPIL POSITION DIFFERENCE");
    console.log("="*60);
    
    if (!pupils[0] || !pupils[1]) {
        console.log("❌ Pupil tidak terdeteksi pada salah satu atau kedua mata");
        console.log("Left pupil:", pupils[0]);
        console.log("Right pupil:", pupils[1]);
        return null;
    }
    
    let left_pupil_x = pupils[0][0];
    let right_pupil_x = pupils[1][0];
    
    let left_eye_w = eyes[0][2];
    let right_eye_w = eyes[1][2];
    
    let left_normalized = left_pupil_x / left_eye_w;
    let right_normalized = right_pupil_x / right_eye_w;
    
    let dx = left_normalized - right_normalized;
    
    console.log(`Left Normalized X: ${left_normalized.toFixed(3)}`);
    console.log(`Right Normalized X: ${right_normalized.toFixed(3)}`);
    console.log(`Difference (dx): ${dx.toFixed(3)}`);
    
    return { dx, left_normalized, right_normalized };
}

// ========================================
// MAIN ANALYSIS FUNCTION (FIXED)
// ========================================
analyzeBtn.addEventListener("click", () => {
    if (!cvReady || !cascadesReady || !faceCascade || !eyeCascade) {
        alert("⏳ Tunggu OpenCV dan cascades siap.");
        return;
    }

    loadingDiv.style.display = "block";
    resultBox.style.display = "none";

    setTimeout(() => {
        try {
            runAnalysis();
        } catch (error) {
            console.error("Analysis error:", error);
            showResult(`❌ Error: ${error.message}`);
            loadingDiv.style.display = "none";
        }
    }, 200);
});

function runAnalysis() {
    // Load image from preview
    let img = filePreview;
    
    // Create canvas to get exact image dimensions
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    
    // Wait for image to load
    img.onload = function() {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        
        // Now process with OpenCV
        processImageWithOpenCV(canvas);
    };
    
    // If image already loaded
    if (img.complete) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        processImageWithOpenCV(canvas);
    }
}

function processImageWithOpenCV(canvas) {
    try {
        // Read image from canvas
        let src = cv.imread(canvas);
        
        console.log("\n" + "="*60);
        console.log("IMAGE PROCESSING START");
        console.log("="*60);
        console.log("Original image size:", src.cols, "x", src.rows, "channels:", src.channels());
        
        // Convert to grayscale (FIX: handle RGBA properly)
        let gray = new cv.Mat();
        
        if (src.channels() === 4) {
            // RGBA to GRAY
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        } else if (src.channels() === 3) {
            // RGB to GRAY
            cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
        } else {
            // Already grayscale
            src.copyTo(gray);
        }
        
        console.log("Gray image size:", gray.cols, "x", gray.rows);
        
        // Step 1: Face Detection
        let [face_box, face_gray] = detect_face_improved(gray);
        
        if (!face_box || !face_gray) {
            showResult("❌ Wajah tidak terdeteksi. Coba gambar dengan wajah yang lebih jelas.");
            cleanup(src, gray);
            return;
        }
        
        console.log("Face gray size:", face_gray.cols, "x", face_gray.rows);
        
        // Step 2: Eye Detection
        let [eyes_in_face, _] = improved_eye_detection(face_gray);
        
        if (eyes_in_face.length < 2) {
            showResult("❌ Kurang dari 2 mata terdeteksi.");
            cleanup(src, gray, face_gray);
            return;
        }
        
        let eyes = eyes_in_face;
        
        // Step 3: Extract Eye ROIs
        console.log("\n" + "="*60);
        console.log("EXTRACT EYE REGIONS");
        console.log("="*60);
        
        // Left eye
        let left_eye_rect = new cv.Rect(eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]);
        let left_eye = face_gray.roi(left_eye_rect);
        console.log("Left eye size:", left_eye.cols, "x", left_eye.rows);
        
        // Right eye
        let right_eye_rect = new cv.Rect(eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]);
        let right_eye = face_gray.roi(right_eye_rect);
        console.log("Right eye size:", right_eye.cols, "x", right_eye.rows);
        
        // Step 4: Pupil Detection
        console.log("\n" + "="*60);
        console.log("PUPIL DETECTION");
        console.log("="*60);
        
        let left_pupil = detect_pupil_hough_simple(left_eye, "Left Eye");
        let right_pupil = detect_pupil_hough_simple(right_eye, "Right Eye");
        
        let pupils = [left_pupil, right_pupil];
        
        // Step 5: Calculate Difference
        let result = calculate_pupil_position_difference(pupils, eyes);
        
        if (result === null) {
            showResult("❌ Tidak dapat menghitung perbedaan posisi pupil. Pupil mungkin tidak terdeteksi.");
            
            // Visualize anyway
            visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
            visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
            
            cleanup(src, gray, face_gray, left_eye, right_eye);
            return;
        }
        
        let { dx, left_normalized, right_normalized } = result;
        
        // Step 6: Display Results
        console.log("\n" + "="*60);
        console.log("FINAL RESULTS");
        console.log("="*60);
        
        // Visualize
        visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
        visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
        
        // Interpretation
        let interpretation = "";
        if (dx < -0.05) {
            interpretation = "Kemungkinan ESOTROPIA (mata menyerong ke dalam)";
        } else if (dx > 0.05) {
            interpretation = "Kemungkinan EXOTROPIA (mata menyerong ke luar)";
        } else {
            interpretation = "Kemungkinan NORMAL";
        }
        
        let htmlOutput = `
            <div class="result-item"><b>Left Pupil:</b> ${left_pupil ? `(x: ${left_pupil[0]}, y: ${left_pupil[1]}, r: ${left_pupil[2]})` : 'Tidak terdeteksi'}</div>
            <div class="result-item"><b>Right Pupil:</b> ${right_pupil ? `(x: ${right_pupil[0]}, y: ${right_pupil[1]}, r: ${right_pupil[2]})` : 'Tidak terdeteksi'}</div>
            <div class="result-item"><b>Left Eye Normalized X:</b> ${left_normalized.toFixed(3)}</div>
            <div class="result-item"><b>Right Eye Normalized X:</b> ${right_normalized.toFixed(3)}</div>
            <div class="result-item"><b>Difference (dx):</b> ${dx.toFixed(3)}</div>
            <div class="result-item"><b>Average Position:</b> ${((left_normalized + right_normalized) / 2).toFixed(3)}</div>
            
            <div class="result-item" style="margin-top: 15px; padding: 10px; background: #e8f4fd; border-left: 4px solid #007bff;">
                <b>💡 Interpretasi:</b><br>
                ${interpretation}<br>
                <small style="color: #666;">(Threshold: dx < -0.05 = Esotropia, dx > 0.05 = Exotropia)</small>
            </div>
            
            <div class="result-item" style="margin-top: 15px; font-style: italic; color: #007bff;">
                Nilai dx untuk gambar ini: ${dx.toFixed(6)}
            </div>
        `;
        
        showResult(htmlOutput);
        
        // Cleanup
        cleanup(src, gray, face_gray, left_eye, right_eye);
        
    } catch (error) {
        console.error("Error in processImageWithOpenCV:", error);
        showResult(`❌ Error: ${error.message}`);
    }
}

// ========================================
// VISUALIZATION FUNCTION
// ========================================
function visualize_eye_result(eye_region, pupil, canvasId, eye_name) {
    let output = new cv.Mat();
    cv.cvtColor(eye_region, output, cv.COLOR_GRAY2RGB);
    
    if (pupil) {
        let [x, y, r] = pupil;
        cv.circle(output, new cv.Point(x, y), r, [0, 255, 0, 255], 2);
        cv.circle(output, new cv.Point(x, y), 2, [255, 0, 0, 255], 3);
    }
    
    let canvas = document.getElementById(canvasId);
    
    // Set canvas size to match image
    canvas.width = eye_region.cols;
    canvas.height = eye_region.rows;
    
    cv.imshow(canvas, output);
    output.delete();
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
                // Ignore delete errors
            }
        }
    });
}

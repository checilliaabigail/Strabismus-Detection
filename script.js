let faceCascade, eyeCascade;
let cvReady = false;
let cascadesReady = false;

// ===== PROGRESS BAR FUNCTIONS =====
function updateProgress(percentage, stepNumber) {
    const progressBar = document.getElementById('progressBarFill');
    const progressContainer = document.getElementById('progressContainer');
    
    // Show progress container
    progressContainer.style.display = 'block';
    
    // Update progress bar
    progressBar.style.width = percentage + '%';
    progressBar.textContent = percentage + '%';
    
    // Update step indicators
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById('step' + i);
        if (i < stepNumber) {
            step.className = 'progress-step completed';
        } else if (i === stepNumber) {
            step.className = 'progress-step active';
        } else {
            step.className = 'progress-step';
        }
    }
}

function resetProgress() {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBarFill');
    
    progressContainer.style.display = 'none';
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    
    for (let i = 1; i <= 4; i++) {
        document.getElementById('step' + i).className = 'progress-step';
    }
}
// ===== END PROGRESS BAR FUNCTIONS =====

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
    
    // Update progress: Face Detection starting
    updateProgress(10, 1);
    
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
    
    // Update progress: Face Detection completed
    updateProgress(25, 1);
    
    // Crop setengah atas wajah (upper half) seperti di Python
    let face_roi = new cv.Rect(fx, fy, fw, Math.floor(fh / 2));
    let face_gray = gray.roi(face_roi);
    
    faces.delete();
    return [best_face, face_gray];
}

// ========================================
// HELPER: EXPAND EYE BOX
// ========================================
function expand_eye_box(eye, img_shape, scale_w = 1.35, scale_h = 1.2) {
    /**
     * Perluas bounding box mata agar mencakup ujung ke ujung mata
     * @param {Array} eye - [x, y, w, h]
     * @param {Object} img_shape - {height, width}
     * @param {Number} scale_w - Scale untuk width (default 1.35)
     * @param {Number} scale_h - Scale untuk height (default 1.2)
     * @returns {Array} - [x_new, y_new, w_new, h_new]
     */
    let [x, y, w, h] = eye;
    let cx = x + Math.floor(w / 2);
    let cy = y + Math.floor(h / 2);
    
    let new_w = Math.floor(w * scale_w);
    let new_h = Math.floor(h * scale_h);
    
    let x_new = Math.max(0, cx - Math.floor(new_w / 2));
    let y_new = Math.max(0, cy - Math.floor(new_h / 2));
    
    // Pastikan tidak melewati batas gambar
    x_new = Math.min(x_new, img_shape.width - new_w);
    y_new = Math.min(y_new, img_shape.height - new_h);
    
    return [x_new, y_new, new_w, new_h];
}

// ========================================
// IMPROVED EYE DETECTION (FINAL VERSION)
// ========================================
function improved_eye_detection(face_gray) {
    console.log("\n" + "="*60);
    console.log("EYE DETECTION");
    console.log("="*60);
    
    // Update progress: Eye Detection starting
    updateProgress(30, 2);
    
    const H = face_gray.rows;
    const W = face_gray.cols;
    
    // Parameter yang lebih fokus dan stabil
    const scaleFactors = [1.02, 1.05, 1.1];
    const minNeighbors_list = [2, 3];
    const minSizes = [
        new cv.Size(15, 15),
        new cv.Size(20, 20),
        new cv.Size(25, 25)
    ];
    
    let best_pair = null;
    let best_score = -1;
    let best_params = null;
    
    for (let sf of scaleFactors) {
        for (let mn of minNeighbors_list) {
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                
                eyeCascade.detectMultiScale(face_gray, eyes, sf, mn, 0, ms);
                
                if (eyes.size() < 2) {
                    eyes.delete();
                    continue;
                }
                
                // Konversi ke array dengan pusat dan area
                let candidates = [];
                for (let i = 0; i < eyes.size(); i++) {
                    let eye = eyes.get(i);
                    let x = eye.x;
                    let y = eye.y;
                    let w = eye.width;
                    let h = eye.height;
                    let cx = x + w / 2;
                    let cy = y + h / 2;
                    let area = w * h;
                    
                    candidates.push({
                        x: x, y: y, w: w, h: h,
                        cx: cx, cy: cy, area: area
                    });
                }
                
                // Coba semua pasangan mata
                for (let i = 0; i < candidates.length; i++) {
                    for (let j = i + 1; j < candidates.length; j++) {
                        let e1 = candidates[i];
                        let e2 = candidates[j];
                        
                        // Constraint 1: Horizontal separation (harus kiri-kanan)
                        if (Math.abs(e1.cx - e2.cx) < 0.25 * W) {
                            continue;
                        }
                        
                        // Constraint 2: Vertical alignment (harus sejajar secara vertikal)
                        if (Math.abs(e1.cy - e2.cy) > 0.2 * H) {
                            continue;
                        }
                        
                        // Constraint 3: Size similarity (ukuran harus mirip)
                        let area_ratio = Math.min(e1.area, e2.area) / Math.max(e1.area, e2.area);
                        if (area_ratio < 0.5) {
                            continue;
                        }
                        
                        // Constraint 4: Non-overlap check (mata tidak boleh overlap)
                        let overlap_x = Math.max(0, Math.min(e1.x + e1.w, e2.x + e2.w) - Math.max(e1.x, e2.x));
                        let overlap_y = Math.max(0, Math.min(e1.y + e1.h, e2.y + e2.h) - Math.max(e1.y, e2.y));
                        if (overlap_x > 0 && overlap_y > 0) {
                            continue;
                        }
                        
                        // Calculate score
                        let horizontal_sep = Math.abs(e1.cx - e2.cx) / W;
                        let vertical_alignment = 1 - Math.abs(e1.cy - e2.cy) / H;
                        let combined_area = (e1.area + e2.area) / (W * H);
                        
                        let score = (
                            horizontal_sep * 0.40 +
                            vertical_alignment * 0.35 +
                            area_ratio * 0.15 +
                            combined_area * 0.10
                        );
                        
                        if (score > best_score) {
                            best_score = score;
                            
                            // Sort left to right
                            if (e1.cx < e2.cx) {
                                best_pair = [e1, e2];
                            } else {
                                best_pair = [e2, e1];
                            }
                            
                            best_params = {sf, mn, minSize: ms.width};
                        }
                    }
                }
                
                eyes.delete();
            }
        }
    }
    
    if (best_pair === null) {
        console.log("❌ Tidak ada pasangan mata valid");
        return [[], null];
    }
    
    console.log(`✅ Best params: sf=${best_params.sf}, mn=${best_params.mn}, minSize=${best_params.minSize}`);
    console.log(`   Score: ${best_score.toFixed(3)}`);
    
    // Expand eye boxes
    let expanded_eyes = [];
    for (let eye of best_pair) {
        let expanded = expand_eye_box(
            [eye.x, eye.y, eye.w, eye.h],
            {height: H, width: W},
            1.35,
            1.2
        );
        expanded_eyes.push(expanded);
        console.log(`   Expanded: (${expanded[0]}, ${expanded[1]}, ${expanded[2]}, ${expanded[3]})`);
    }
    
    // Update progress: Eye Detection completed
    updateProgress(50, 2);
    
    return [expanded_eyes, best_params];
}

// ========================================
// PUPIL DETECTION (SIMPLIFIED)
// ========================================
function detect_pupil_hough_simple(eye_region, eye_name = "Eye") {
    console.log(`\nDeteksi Pupil: ${eye_name}`);
    
    if (!eye_region || eye_region.empty()) {
        console.log("❌ Eye region kosong");
        return null;
    }
    
    let h = eye_region.rows;
    let w = eye_region.cols;
    
    console.log(`  Ukuran: ${w} x ${h} px`);
    
    // Preprocessing
    let blurred = new cv.Mat();
    cv.GaussianBlur(eye_region, blurred, new cv.Size(5, 5), 0);
    
    let clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    let enhanced = new cv.Mat();
    clahe.apply(blurred, enhanced);
    
    // Estimate pupil size
    let min_radius = Math.max(5, Math.floor(w * 0.12));
    let max_radius = Math.min(Math.floor(w * 0.25), 40);
    
    console.log(`  Radius range: ${min_radius}-${max_radius} px`);
    
    // Hough Circle Detection
    let circles = new cv.Mat();
    let params_list = [
        [1.15, 48, 78, 17],
        [1.1, 50, 80, 20],
        [1.14, 52, 75, 18],
        [1.1, 60, 80, 17]
    ];
    
    let best_circle = null;
    let best_score = -1;
    
    for (let [dp, minDist, param1, param2] of params_list) {
        circles = new cv.Mat();
        
        try {
            cv.HoughCircles(
                enhanced,
                circles,
                cv.HOUGH_GRADIENT,
                dp,
                minDist,
                param1,
                param2,
                min_radius,
                max_radius
            );
            
            if (circles.cols > 0) {
                for (let i = 0; i < circles.cols; i++) {
                    let x = circles.data32F[i * 3];
                    let y = circles.data32F[i * 3 + 1];
                    let r = circles.data32F[i * 3 + 2];
                    
                    // Validate bounds
                    if (x - r < 0 || x + r > w || y - r < 0 || y + r > h) {
                        continue;
                    }
                    
                    // Validate margin
                    let edge_margin = 5;
                    if (x < edge_margin || x > w - edge_margin || 
                        y < edge_margin || y > h - edge_margin) {
                        continue;
                    }
                    
                    // Calculate intensity
                    let mask = cv.Mat.zeros(eye_region.rows, eye_region.cols, cv.CV_8UC1);
                    cv.circle(mask, new cv.Point(x, y), r, [255, 255, 255, 255], -1);
                    
                    let mean_intensity = cv.mean(eye_region, mask)[0];
                    
                    mask.delete();
                    
                    // Skip if too bright
                    if (mean_intensity > 100) {
                        continue;
                    }
                    
                    // Calculate score
                    let center_x = w / 2;
                    let center_y = h / 2;
                    let dist = Math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2);
                    let center_score = 1.0 / (1.0 + dist / (w * 0.25));
                    let intensity_score = Math.max(0, 1.0 - mean_intensity / 80.0);
                    
                    let score = center_score * 0.5 + intensity_score * 0.5;
                    
                    if (score > best_score) {
                        best_score = score;
                        best_circle = [Math.round(x), Math.round(y), Math.round(r)];
                    }
                }
            }
        } catch (e) {
            // Continue to next parameter set
        }
        
        circles.delete();
    }
    
    blurred.delete();
    enhanced.delete();
    
    if (best_circle) {
        console.log(`  ✅ Pupil found: (${best_circle[0]}, ${best_circle[1]}), r=${best_circle[2]}, score=${best_score.toFixed(3)}`);
        return best_circle;
    } else {
        console.log(`  ❌ No pupil detected`);
        return null;
    }
}

// ========================================
// CALCULATE PUPIL POSITION DIFFERENCE
// ========================================
function calculate_pupil_position_difference(pupils, eyes) {
    console.log("\n" + "="*60);
    console.log("CALCULATING PUPIL POSITION DIFFERENCE");
    console.log("="*60);
    
    let left_pupil = pupils[0];
    let right_pupil = pupils[1];
    let left_eye = eyes[0];
    let right_eye = eyes[1];
    
    if (!left_pupil || !right_pupil) {
        console.log("❌ Salah satu pupil tidak terdeteksi");
        return null;
    }
    
    // Normalize pupil position
    let left_normalized = left_pupil[0] / left_eye[2];
    let right_normalized = right_pupil[0] / right_eye[2];
    
    // Calculate difference
    let dx = left_normalized - right_normalized;
    
    console.log(`Left pupil normalized X: ${left_normalized.toFixed(3)}`);
    console.log(`Right pupil normalized X: ${right_normalized.toFixed(3)}`);
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
    resetProgress();

    setTimeout(() => {
        try {
            runAnalysis();
        } catch (error) {
            console.error("Analysis error:", error);
            showResult(`❌ Error: ${error.message}`);
            loadingDiv.style.display = "none";
            resetProgress();
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
        
        // Update progress: Starting
        updateProgress(5, 1);
        
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
            resetProgress();
            return;
        }
        
        console.log("Face gray size:", face_gray.cols, "x", face_gray.rows);
        
        // Step 2: Eye Detection
        let [eyes_in_face, _] = improved_eye_detection(face_gray);
        
        if (eyes_in_face.length < 2) {
            showResult("❌ Kurang dari 2 mata terdeteksi.");
            cleanup(src, gray, face_gray);
            resetProgress();
            return;
        }
        
        let eyes = eyes_in_face;
        
        // Step 3: Extract Eye ROIs
        console.log("\n" + "="*60);
        console.log("EXTRACT EYE REGIONS");
        console.log("="*60);
        
        // Update progress: Pupil Detection starting
        updateProgress(55, 3);
        
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
        
        // Update progress: Left pupil detected
        updateProgress(70, 3);
        
        let right_pupil = detect_pupil_hough_simple(right_eye, "Right Eye");
        
        // Update progress: Right pupil detected
        updateProgress(85, 3);
        
        let pupils = [left_pupil, right_pupil];
        
        // Step 5: Calculate Difference
        // Update progress: Calculation starting
        updateProgress(90, 4);
        
        let result = calculate_pupil_position_difference(pupils, eyes);
        
        if (result === null) {
            showResult("❌ Tidak dapat menghitung perbedaan posisi pupil. Pupil mungkin tidak terdeteksi.");
            
            // Visualize anyway
            visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
            visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
            
            cleanup(src, gray, face_gray, left_eye, right_eye);
            resetProgress();
            return;
        }
        
        let { dx, left_normalized, right_normalized } = result;
        
        // Step 6: Display Results
        console.log("\n" + "="*60);
        console.log("FINAL RESULTS");
        console.log("="*60);
        
        // Update progress: Finalizing
        updateProgress(95, 4);
        
        // Visualize
        visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
        visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
        
        // Interpretation
        let interpretation = "";
        if (dx < -0.0586) {
            interpretation = "EXOTROPIA (mata menyerong ke luar)";
        } else if (dx < 0.1115) {
            interpretation = "NORMAL";
        } else {
            interpretation = "ESOTROPIA";
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
        
        // Update progress: Complete!
        updateProgress(100, 4);
        
        showResult(htmlOutput);
        
        // Cleanup
        cleanup(src, gray, face_gray, left_eye, right_eye);
        
        // Hide progress bar after 2 seconds
        setTimeout(() => {
            resetProgress();
        }, 2000);
        
    } catch (error) {
        console.error("Error in processImageWithOpenCV:", error);
        showResult(`❌ Error: ${error.message}`);
        resetProgress();
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

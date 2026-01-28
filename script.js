let faceCascade, eyeCascade;
let cvReady = false;
let cascadesReady = false;

// Inisiasi OpenCV
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
// Face Detection dengan Haar Cascade 
// ========================================
function detect_face_improved(gray) {
    console.log("\n" + "="*60);
    console.log("FACE DETECTION");
    console.log("="*60);
    
    let faces = new cv.RectVector();
    let minSize = new cv.Size(80, 80);
    
    // Coba beberapa parameter
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
                
                // scoring
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
    
    // Crop setengah atas wajah (upper half)
    let face_roi = new cv.Rect(fx, fy, fw, Math.floor(fh / 2));
    let face_gray = gray.roi(face_roi);
    
    faces.delete();
    return [best_face, face_gray];
}

// ========================================
// Bantuan dengan perluasan eye box
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
// Eye Detection dengan Haar Cascade
// ========================================
function improved_eye_detection(face_gray) {
    console.log("\n" + "="*60);
    console.log("EYE DETECTION");
    console.log("="*60);
    
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
                        
                        // Constraint 2: Vertical alignment (harus sejajar)
                        if (Math.abs(e1.cy - e2.cy) > 0.15 * H) {
                            continue;
                        }
                        
                        // Score = total area
                        let score = e1.area + e2.area;
                        
                        if (score > best_score) {
                            best_score = score;
                            best_pair = [
                                [e1.x, e1.y, e1.w, e1.h],
                                [e2.x, e2.y, e2.w, e2.h]
                            ];
                            best_params = {sf: sf, mn: mn, ms: ms};
                        }
                    }
                }
                
                eyes.delete();
            }
        }
    }
    
    if (best_pair === null) {
        console.log("❌ Gagal mendeteksi pasangan mata");
        return [[], null];
    }
    
    // Sort kiri → kanan
    best_pair.sort((a, b) => a[0] - b[0]);
    
    console.log("✅ Mata terdeteksi dengan stabil");
    console.log(`   Parameter terbaik: scaleFactor=${best_params.sf}, minNeighbors=${best_params.mn}, minSize=${best_params.ms.width}x${best_params.ms.height}`);
    
    let img_shape = {height: H, width: W};
    let left_eye_expanded = expand_eye_box(best_pair[0], img_shape);
    let right_eye_expanded = expand_eye_box(best_pair[1], img_shape);
    
    let final_eyes = [left_eye_expanded, right_eye_expanded];
    
    console.log(`   Original boxes: Kiri=${best_pair[0]}, Kanan=${best_pair[1]}`);
    console.log(`   Expanded boxes: Kiri=${final_eyes[0]}, Kanan=${final_eyes[1]}`);
    
    return [final_eyes, best_params];
}

// ========================================
// Pupil Detection dengan Hough Circle Detection
// ========================================
function detect_pupil_hough_simple(eye_gray, eye_name = "Eye") {
    console.log(`\n${eye_name} PUPIL DETECTION`);
    
    if (!eye_gray || eye_gray.rows === 0 || eye_gray.cols === 0) {
        console.log("❌ Region mata kosong");
        return null;
    }
    
    let h = eye_gray.rows, w = eye_gray.cols;
    console.log(`Ukuran: ${w}x${h}`);
    
    
    let blurred = new cv.Mat();
    cv.GaussianBlur(eye_gray, blurred, new cv.Size(7, 7), 2, 2, cv.BORDER_DEFAULT);
    let min_radius = Math.max(5, Math.floor(w * 0.12));
    let max_radius = Math.min(Math.floor(w * 0.25), 40);
    
    console.log(`Estimasi radius pupil: ${min_radius}-${max_radius} px`);
    
    let param_sets = [
        // Parameter exotropia
        {dp: 1.15, minDist: 48, param1: 78, param2: 17, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 50, param1: 80, param2: 20, minR: min_radius, maxR: max_radius},
        {dp: 1.2, minDist: 45, param1: 75, param2: 15, minR: min_radius, maxR: max_radius},
        
        // Parameter normal
        {dp: 1.14, minDist: 52, param1: 75, param2: 18, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 50, param1: 80, param2: 18, minR: min_radius, maxR: max_radius},
        {dp: 1.2, minDist: 55, param1: 70, param2: 18, minR: min_radius, maxR: max_radius},
        
        // Parameter esotropia
        {dp: 1.13, minDist: 67, param1: 80, param2: 16, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 70, param1: 80, param2: 15, minR: min_radius, maxR: max_radius},
        {dp: 1.2, minDist: 65, param1: 80, param2: 17, minR: min_radius, maxR: max_radius},
        
        // GABUNGAN
        {dp: 1.14, minDist: 56, param1: 78, param2: 17, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 60, param1: 80, param2: 17, minR: min_radius, maxR: max_radius},
        {dp: 1.2, minDist: 55, param1: 75, param2: 17, minR: min_radius, maxR: max_radius},
        
        // Ekstrem dari dataset
        {dp: 1.1, minDist: 20, param1: 80, param2: 20, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 70, param1: 90, param2: 15, minR: min_radius, maxR: max_radius},
        {dp: 1.2, minDist: 60, param1: 60, param2: 10, minR: min_radius, maxR: max_radius},
        {dp: 1.1, minDist: 50, param1: 110, param2: 27, minR: min_radius, maxR: max_radius}
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
                
                
                for (let i = 0; i < circles.cols; i++) {
                    let x = Math.round(circles.data32F[i * 3]);
                    let y = Math.round(circles.data32F[i * 3 + 1]);
                    let r = Math.round(circles.data32F[i * 3 + 2]);
                    
                    // Validasi posisi (dalam bounds)
                    if (x - r > 0 && x + r < w && y - r > 0 && y + r < h) {
                        if (r >= min_radius && r <= max_radius) {
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
        
        for (let circle of all_circles) {
            let x = circle[0], y = circle[1], r = circle[2];
            
            let mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);
            cv.circle(mask, new cv.Point(x, y), r, new cv.Scalar(255), -1);
            let mean_intensity = cv.mean(eye_gray, mask)[0];
            mask.delete();
            
            // ===========
            // SCORING
            // ===========
            
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
            
            // 3. Intensity check - pupil harus gelap. Kalau tertalu terang di skip
            if (mean_intensity > 100) {
                continue;
            }
            
            // 4. Total score (70% center, 30% radius)
            let total_score = center_score * 0.7 + radius_score * 0.3;
            
            scored_circles.push({
                score: total_score,
                circle: [x, y, r],
                intensity: mean_intensity
            });
            
            console.log(`  Circle: (${x}, ${y}, r=${r}) intensity=${mean_intensity.toFixed(1)}, score=${total_score.toFixed(3)}`);
        }
        
        // Sort dari yang paling bagus
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
// Perhitungan perbedaan posisi pupil
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
// Fungsi Analisis Utama
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
    // Load image dari  preview
    let img = filePreview;
    
    // Buat canvas untuk mendapatkan dimensi citra
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    
    // Tunggu citra sampai ter load
    img.onload = function() {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        
        // Proses dengan OpenCV
        processImageWithOpenCV(canvas);
    };
    
    // Kalau image sudah terload
    if (img.complete) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        processImageWithOpenCV(canvas);
    }
}

function processImageWithOpenCV(canvas) {
    try {
        // Baca dari canvas
        let src = cv.imread(canvas);
        
        console.log("\n" + "="*60);
        console.log("IMAGE PROCESSING START");
        console.log("="*60);
        console.log("Original image size:", src.cols, "x", src.rows, "channels:", src.channels());
        
        // Preprocessing ke Grayscale
        let gray = new cv.Mat();
        
        if (src.channels() === 4) {
            // RGBA to GRAY
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        } else if (src.channels() === 3) {
            // RGB to GRAY
            cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
        } else {
            // Sudah grayscale
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
        
        // Step 3: Extract ROI Mata
        console.log("\n" + "="*60);
        console.log("EXTRACT EYE REGIONS");
        console.log("="*60);
        
        // Mata kiri
        let left_eye_rect = new cv.Rect(eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]);
        let left_eye = face_gray.roi(left_eye_rect);
        console.log("Left eye size:", left_eye.cols, "x", left_eye.rows);
        
        // Mata kanan
        let right_eye_rect = new cv.Rect(eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]);
        let right_eye = face_gray.roi(right_eye_rect);
        console.log("Right eye size:", right_eye.cols, "x", right_eye.rows);
        
        // Step 4: Deteksi pupil
        console.log("\n" + "="*60);
        console.log("PUPIL DETECTION");
        console.log("="*60);
        
        let left_pupil = detect_pupil_hough_simple(left_eye, "Left Eye");
        let right_pupil = detect_pupil_hough_simple(right_eye, "Right Eye");
        
        let pupils = [left_pupil, right_pupil];
        
        // Step 5: Perhitungan perbedaan posisi pupil
        let result = calculate_pupil_position_difference(pupils, eyes);
        
        if (result === null) {
            showResult("❌ Tidak dapat menghitung perbedaan posisi pupil. Pupil mungkin tidak terdeteksi.");
            
            // Visualisasi meskipun tidak ada yang terdeteksi
            visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
            visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
            
            cleanup(src, gray, face_gray, left_eye, right_eye);
            return;
        }
        
        let { dx, left_normalized, right_normalized } = result;
        
        // Step 6: Tampilkan hasil
        console.log("\n" + "="*60);
        console.log("FINAL RESULTS");
        console.log("="*60);
        
        // Visualisasi
        visualize_eye_result(left_eye, left_pupil, "leftEyeCanvas", "Left Eye");
        visualize_eye_result(right_eye, right_pupil, "rightEyeCanvas", "Right Eye");
        
        // Logika Interpretasi threshold
        let interpretation = "";
        if (dx < -0.0671) {
            interpretation = "EXOTROPIA (mata menyerong ke luar)";
        } else if (dx < 0.1019) {
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
                <small style="color: #666;">(Threshold: dx < -0.0671 = Esotropia, dx > 0.1019 = Exotropia)</small>
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

// ===================
// Untuk Visualisasi
// ===================
function visualize_eye_result(eye_region, pupil, canvasId, eye_name) {
    let output = new cv.Mat();
    cv.cvtColor(eye_region, output, cv.COLOR_GRAY2RGB);
    
    if (pupil) {
        let [x, y, r] = pupil;
        cv.circle(output, new cv.Point(x, y), r, [0, 255, 0, 255], 2);
        cv.circle(output, new cv.Point(x, y), 2, [255, 0, 0, 255], 3);
    }
    
    let canvas = document.getElementById(canvasId);
    
    // Set ukuran canvas untuk sama dengan citra
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
            }
        }
    });
}

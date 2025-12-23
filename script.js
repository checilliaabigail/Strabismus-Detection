// Global variables
let cv = null;
let faceCascade = null;
let eyeCascade = null;
let isOpenCvReady = false;

// Log functions
function addLog(message, type = 'info') {
    const logContainer = document.getElementById('logContainer');
    const logItem = document.createElement('div');
    logItem.className = `log-item ${type}`;
    logItem.textContent = message;
    logContainer.appendChild(logItem);
    console.log(message);
}

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
}

function showLoader(show) {
    document.getElementById('loader').classList.toggle('active', show);
}

// OpenCV ready callback
function onOpenCvReady() {
    cv = window.cv;
    isOpenCvReady = true;
    console.log('OpenCV.js is ready!');
    showStatus('✅ OpenCV.js berhasil dimuat. Silakan upload gambar.', 'success');
    
    // Load Haar Cascades
    loadHaarCascades();
}

// Load Haar Cascade files
function loadHaarCascades() {
    // Load face cascade
    let faceCascadeFile = 'haarcascade_frontalface_default.xml';
    let eyeCascadeFile = 'haarcascade_eye.xml';
    
    createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        faceCascade = new cv.CascadeClassifier();
        faceCascade.load(faceCascadeFile);
        addLog('✅ Face cascade loaded', 'success');
    });
    
    createFileFromUrl(eyeCascadeFile, eyeCascadeFile, () => {
        eyeCascade = new cv.CascadeClassifier();
        eyeCascade.load(eyeCascadeFile);
        addLog('✅ Eye cascade loaded', 'success');
    });
}

function createFileFromUrl(path, url, callback) {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function() {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                addLog('❌ Failed to load ' + path, 'error');
            }
        }
    };
    request.send();
}

// File input handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (!isOpenCvReady) {
        showStatus('⏳ Tunggu OpenCV.js selesai loading...', 'warning');
        return;
    }
    
    const file = e.target.files[0];
    if (!file) return;
    
    // Clear previous results
    document.getElementById('logContainer').innerHTML = '';
    document.getElementById('results').style.display = 'none';
    document.getElementById('resultBox').style.display = 'none';
    
    showLoader(true);
    showStatus('📤 Memuat gambar...', 'info');
    
    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            processImage(img);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

// Main image processing function
function processImage(img) {
    try {
        addLog('🔄 Memulai proses deteksi...', 'info');
        
        // Show results section
        document.getElementById('results').style.display = 'block';
        
        // Step 1: Load and display original image
        let src = cv.imread(img);
        addLog(`📊 Ukuran gambar: ${src.cols}x${src.rows}`, 'info');
        cv.imshow('originalCanvas', src);
        
        // Step 2: Convert to grayscale
        let gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.imshow('grayCanvas', gray);
        addLog('✅ Konversi ke grayscale berhasil', 'success');
        
        // Step 3: Face detection with automatic parameter tuning
        addLog('🔍 Mencari wajah dengan berbagai kombinasi parameter...', 'info');
        const faceResult = improvedFaceDetection(gray);
        
        if (!faceResult) {
            showStatus('❌ Tidak dapat mendeteksi wajah pada gambar', 'error');
            addLog('❌ Tidak ada wajah terdeteksi dengan semua kombinasi parameter', 'error');
            showLoader(false);
            src.delete();
            gray.delete();
            return;
        }
        
        const { face, faceGray, params: faceParams } = faceResult;
        addLog(`✅ Wajah terdeteksi!`, 'success');
        addLog(`   Parameter terbaik: scaleFactor=${faceParams.scaleFactor}, minNeighbors=${faceParams.minNeighbors}, minSize=${faceParams.minSize}`, 'info');
        addLog(`📊 Info Wajah:`, 'info');
        addLog(`   Posisi: x=${face.x}, y=${face.y}`, 'info');
        addLog(`   Ukuran: width=${face.width}, height=${face.height}`, 'info');
        
        // Display face detection result
        let faceImg = src.clone();
        cv.rectangle(faceImg, new cv.Point(face.x, face.y), 
                     new cv.Point(face.x + face.width, face.y + face.height),
                     [0, 255, 0, 255], 3);
        cv.imshow('faceCanvas', faceImg);
        faceImg.delete();
        
        // Display cropped face
        cv.imshow('croppedFaceCanvas', faceGray);
        
        // Step 4: Eye detection
        addLog('🔍 Mencari mata dengan berbagai kombinasi parameter...', 'info');
        const eyeResult = improvedEyeDetection(faceGray);
        
        if (!eyeResult || eyeResult.eyes.length < 2) {
            showStatus('❌ Tidak dapat mendeteksi 2 mata pada wajah', 'error');
            addLog('❌ Tidak dapat mendeteksi 2 mata', 'error');
            showLoader(false);
            src.delete();
            gray.delete();
            faceGray.delete();
            return;
        }
        
        const { eyes, params: eyeParams } = eyeResult;
        addLog(`✅ Kedua mata terdeteksi!`, 'success');
        addLog(`   Parameter terbaik: scaleFactor=${eyeParams.scaleFactor}, minNeighbors=${eyeParams.minNeighbors}, minSize=${eyeParams.minSize}`, 'info');
        
        // Sort eyes from left to right
        eyes.sort((a, b) => a.x - b.x);
        addLog(`📊 Info Mata:`, 'info');
        addLog(`   Mata kiri: (${eyes[0].x}, ${eyes[0].y}, ${eyes[0].width}, ${eyes[0].height})`, 'info');
        addLog(`   Mata kanan: (${eyes[1].x}, ${eyes[1].y}, ${eyes[1].width}, ${eyes[1].height})`, 'info');
        
        // Display eyes on face
        let faceWithEyes = new cv.Mat();
        cv.cvtColor(faceGray, faceWithEyes, cv.COLOR_GRAY2RGBA);
        for (let eye of eyes) {
            cv.rectangle(faceWithEyes, new cv.Point(eye.x, eye.y),
                        new cv.Point(eye.x + eye.width, eye.y + eye.height),
                        [255, 0, 0, 255], 2);
        }
        cv.imshow('eyesCanvas', faceWithEyes);
        faceWithEyes.delete();
        
        // Extract left and right eyes
        let leftEye = faceGray.roi(new cv.Rect(eyes[0].x, eyes[0].y, eyes[0].width, eyes[0].height));
        let rightEye = faceGray.roi(new cv.Rect(eyes[1].x, eyes[1].y, eyes[1].width, eyes[1].height));
        
        cv.imshow('leftEyeCanvas', leftEye);
        cv.imshow('rightEyeCanvas', rightEye);
        
        // Step 5: Pupil detection using Hough Transform
        addLog('🔍 Mendeteksi pupil menggunakan Hough Transform...', 'info');
        
        const leftPupilResult = detectPupilWithFallback(leftEye, 'Left Eye');
        const rightPupilResult = detectPupilWithFallback(rightEye, 'Right Eye');
        
        addLog(`\nConfidence: Left=${leftPupilResult.confidence}, Right=${rightPupilResult.confidence}`, 'info');
        
        // Visualize pupil detection results
        visualizeResult(leftEye, leftPupilResult.circle, 'Left Eye', leftPupilResult.confidence, 'leftPupilCanvas');
        visualizeResult(rightEye, rightPupilResult.circle, 'Right Eye', rightPupilResult.confidence, 'rightPupilCanvas');
        
        // Step 6: Calculate pupil position difference
        addLog('\n' + '='.repeat(60), 'info');
        addLog('PUPIL POSITION DIFFERENCE ANALYSIS', 'info');
        addLog('='.repeat(60), 'info');
        
        const pupils = [leftPupilResult.circle, rightPupilResult.circle];
        const positionResult = calculatePupilPositionDifference(pupils, eyes);
        
        if (positionResult) {
            const { dx, leftNorm, rightNorm } = positionResult;
            
            addLog(`Left Eye Normalized X:  ${leftNorm.toFixed(3)}`, 'info');
            addLog(`Right Eye Normalized X: ${rightNorm.toFixed(3)}`, 'info');
            addLog(`Difference (dx):        ${dx.toFixed(3)}`, 'info');
            addLog(`Average Position:       ${((leftNorm + rightNorm)/2).toFixed(3)}`, 'info');
            
            addLog(`\nNilai dx untuk gambar ini: ${dx.toFixed(6)}`, 'success');
            
            // Interpretation
            let diagnosis = '';
            let diagnosisClass = '';
            
            if (dx < -0.05) {
                diagnosis = 'ESOTROPIA (mata menyerong ke dalam)';
                diagnosisClass = 'error';
                addLog('\n💡 Interpretasi: → Kemungkinan ESOTROPIA (mata menyerong ke dalam)', 'warning');
            } else if (dx > 0.05) {
                diagnosis = 'EXOTROPIA (mata menyerong ke luar)';
                diagnosisClass = 'warning';
                addLog('\n💡 Interpretasi: → Kemungkinan EXOTROPIA (mata menyerong ke luar)', 'warning');
            } else {
                diagnosis = 'NORMAL';
                diagnosisClass = 'success';
                addLog('\n💡 Interpretasi: → Kemungkinan NORMAL', 'success');
            }
            
            // Display final result
            const resultBox = document.getElementById('resultBox');
            const resultDetails = document.getElementById('resultDetails');
            
            resultDetails.innerHTML = `
                <p><strong>Posisi Pupil Mata Kiri (normalized):</strong> ${leftNorm.toFixed(3)}</p>
                <p><strong>Posisi Pupil Mata Kanan (normalized):</strong> ${rightNorm.toFixed(3)}</p>
                <p><strong>Perbedaan Posisi (dx):</strong> ${dx.toFixed(6)}</p>
                <p><strong>Posisi Rata-rata:</strong> ${((leftNorm + rightNorm)/2).toFixed(3)}</p>
                <p><strong>Confidence Deteksi Kiri:</strong> <span class="confidence-badge confidence-${leftPupilResult.confidence}">${leftPupilResult.confidence.toUpperCase()}</span></p>
                <p><strong>Confidence Deteksi Kanan:</strong> <span class="confidence-badge confidence-${rightPupilResult.confidence}">${rightPupilResult.confidence.toUpperCase()}</span></p>
                <div class="diagnosis">${diagnosis}</div>
            `;
            
            resultBox.style.display = 'block';
            showStatus(`✅ Proses selesai! Diagnosis: ${diagnosis}`, diagnosisClass);
            
        } else {
            addLog('Tidak dapat menghitung perbedaan posisi pupil', 'error');
            showStatus('❌ Tidak dapat menghitung perbedaan posisi pupil', 'error');
        }
        
        addLog('='.repeat(60), 'info');
        
        // Cleanup
        src.delete();
        gray.delete();
        faceGray.delete();
        leftEye.delete();
        rightEye.delete();
        
        showLoader(false);
        
    } catch (error) {
        console.error('Error:', error);
        showStatus('❌ Terjadi kesalahan: ' + error.message, 'error');
        addLog('❌ Error: ' + error.message, 'error');
        showLoader(false);
    }
}

// ================================================================================
// IMPROVED FACE DETECTION (sama persis dengan Python)
// ================================================================================
function improvedFaceDetection(gray) {
    // Parameter yang akan dicoba (sama dengan Python)
    const scaleFactors = [1.05, 1.1, 1.15, 1.2];
    const minNeighborsList = [3, 4, 5, 6];
    const minSizes = [[50, 50], [60, 60], [80, 80], [100, 100]];
    
    let bestFace = null;
    let bestScore = -1;
    let bestParams = null;
    
    for (let sf of scaleFactors) {
        for (let mn of minNeighborsList) {
            for (let ms of minSizes) {
                let faces = new cv.RectVector();
                faceCascade.detectMultiScale(
                    gray,
                    faces,
                    sf,          // scaleFactor
                    mn,          // minNeighbors
                    0,           // flags
                    new cv.Size(ms[0], ms[1]),  // minSize
                    new cv.Size(0, 0)           // maxSize
                );
                
                if (faces.size() > 0) {
                    // Ambil wajah dengan area terbesar
                    let facesSorted = [];
                    for (let i = 0; i < faces.size(); i++) {
                        let face = faces.get(i);
                        facesSorted.push({
                            x: face.x,
                            y: face.y,
                            width: face.width,
                            height: face.height,
                            area: face.width * face.height
                        });
                    }
                    facesSorted.sort((a, b) => b.area - a.area);
                    
                    let face = facesSorted[0];
                    
                    // Hitung score berdasarkan area dan rasio aspek
                    let area = face.width * face.height;
                    let aspectRatio = face.width / face.height;
                    
                    // Rasio aspek ideal wajah sekitar 0.75-1.3
                    let aspectPenalty = Math.abs(1.0 - aspectRatio);
                    let score = area * (1 - aspectPenalty * 0.5);
                    
                    if (score > bestScore) {
                        bestScore = score;
                        bestFace = face;
                        bestParams = {
                            scaleFactor: sf,
                            minNeighbors: mn,
                            minSize: `(${ms[0]}, ${ms[1]})`
                        };
                    }
                }
                
                faces.delete();
            }
        }
    }
    
    if (!bestFace) {
        return null;
    }
    
    // Crop wajah (setengah atas untuk area mata)
    let faceGray = gray.roi(new cv.Rect(
        bestFace.x,
        bestFace.y,
        bestFace.width,
        Math.floor(bestFace.height / 2)
    ));
    
    return {
        face: bestFace,
        faceGray: faceGray,
        params: bestParams
    };
}

// ================================================================================
// IMPROVED EYE DETECTION (sama persis dengan Python)
// ================================================================================
function improvedEyeDetection(gray) {
    // Parameter yang lebih sensitif untuk gambar close-up (sama dengan Python)
    const scaleFactors = [1.01, 1.02, 1.05, 1.1];
    const minNeighborsList = [1, 2, 3];
    const minSizes = [[10, 10], [15, 15], [20, 20], [25, 25]];
    
    let bestEyes = null;
    let bestScore = -1;
    let bestParams = null;
    
    for (let sf of scaleFactors) {
        for (let mn of minNeighborsList) {
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                eyeCascade.detectMultiScale(
                    gray,
                    eyes,
                    sf,          // scaleFactor
                    mn,          // minNeighbors
                    0,           // flags
                    new cv.Size(ms[0], ms[1]),  // minSize
                    new cv.Size(0, 0)           // maxSize
                );
                
                // Filter: hanya ambil 2 mata dengan area terbesar
                if (eyes.size() >= 2) {
                    let eyesList = [];
                    for (let i = 0; i < eyes.size(); i++) {
                        let eye = eyes.get(i);
                        eyesList.push({
                            x: eye.x,
                            y: eye.y,
                            width: eye.width,
                            height: eye.height,
                            area: eye.width * eye.height
                        });
                    }
                    eyesList.sort((a, b) => b.area - a.area);
                    
                    let eye1 = eyesList[0];
                    let eye2 = eyesList[1];
                    
                    // Pastikan kedua mata tidak tumpang tindih
                    let dx = Math.min(eye1.x + eye1.width, eye2.x + eye2.width) - Math.max(eye1.x, eye2.x);
                    let dy = Math.min(eye1.y + eye1.height, eye2.y + eye2.height) - Math.max(eye1.y, eye2.y);
                    
                    if (dx <= 0 || dy <= 0) {  // Tidak overlap
                        let score = eye1.area + eye2.area;
                        if (score > bestScore) {
                            bestScore = score;
                            bestEyes = [eye1, eye2];
                            bestParams = {
                                scaleFactor: sf,
                                minNeighbors: mn,
                                minSize: `(${ms[0]}, ${ms[1]})`
                            };
                        }
                    }
                }
                
                eyes.delete();
            }
        }
    }
    
    if (!bestEyes || bestEyes.length < 2) {
        return null;
    }
    
    return {
        eyes: bestEyes,
        params: bestParams
    };
}

// ================================================================================
// HOUGH TRANSFORM PUPIL DETECTION (sama persis dengan Python)
// ================================================================================
function detectPupilImproved(eyeRegion, eyeName = "Eye", debug = false) {
    const strategies = {
        // Strategy 1: Threshold Simple
        'threshold_simple': (eye) => {
            let processed = new cv.Mat();
            cv.threshold(eye, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            return processed;
        },
        
        // Strategy 2: Adaptive Threshold
        'adaptive_threshold': (eye) => {
            let processed = new cv.Mat();
            cv.adaptiveThreshold(eye, processed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 11, 2);
            return processed;
        },
        
        // Strategy 3: Histogram Equalization + Threshold
        'hist_eq': (eye) => {
            let equalized = new cv.Mat();
            cv.equalizeHist(eye, equalized);
            let processed = new cv.Mat();
            cv.threshold(equalized, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            equalized.delete();
            return processed;
        },
        
        // Strategy 4: Gaussian Blur + Threshold
        'blur_threshold': (eye) => {
            let blurred = new cv.Mat();
            cv.GaussianBlur(eye, blurred, new cv.Size(5, 5), 0);
            let processed = new cv.Mat();
            cv.threshold(blurred, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            blurred.delete();
            return processed;
        },
        
        // Strategy 5: Morphological Opening
        'morph_open': (eye) => {
            let processed = new cv.Mat();
            cv.threshold(eye, processed, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
            let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
            cv.morphologyEx(processed, processed, cv.MORPH_OPEN, kernel);
            kernel.delete();
            return processed;
        }
    };
    
    let bestCircle = null;
    let bestScore = -1;
    let bestStrategy = '';
    
    for (let [strategyName, strategyFunc] of Object.entries(strategies)) {
        try {
            let processed = strategyFunc(eyeRegion);
            
            // Hough Circle Detection (parameter sama dengan Python)
            let circles = new cv.Mat();
            cv.HoughCircles(
                processed,
                circles,
                cv.HOUGH_GRADIENT,
                1,                    // dp
                eyeRegion.rows / 8,   // minDist
                100,                  // param1
                15,                   // param2
                Math.floor(eyeRegion.cols * 0.15),  // minRadius
                Math.floor(eyeRegion.cols * 0.4)    // maxRadius
            );
            
            if (circles.cols > 0) {
                // Evaluasi setiap circle
                for (let i = 0; i < circles.cols; i++) {
                    let x = circles.data32F[i * 3];
                    let y = circles.data32F[i * 3 + 1];
                    let r = circles.data32F[i * 3 + 2];
                    
                    // Hitung intensitas rata-rata dalam circle
                    let mask = cv.Mat.zeros(eyeRegion.rows, eyeRegion.cols, cv.CV_8UC1);
                    cv.circle(mask, new cv.Point(x, y), r, [255, 255, 255, 255], -1);
                    let meanIntensity = cv.mean(eyeRegion, mask)[0];
                    mask.delete();
                    
                    // Score: prioritaskan intensitas rendah (gelap)
                    let intensityScore = Math.max(0, 100 - meanIntensity) / 100;
                    let radiusScore = r / eyeRegion.cols;  // radius relatif
                    let centerScore = 1.0 - (Math.abs(x - eyeRegion.cols/2) / (eyeRegion.cols/2));
                    
                    let score = intensityScore * 0.6 + radiusScore * 0.2 + centerScore * 0.2;
                    
                    // Filter: intensitas harus di bawah threshold
                    if (meanIntensity < 100 && score > bestScore) {
                        bestScore = score;
                        bestCircle = [Math.round(x), Math.round(y), Math.round(r)];
                        bestStrategy = strategyName;
                    }
                }
            }
            
            circles.delete();
            processed.delete();
            
        } catch (e) {
            continue;
        }
    }
    
    if (bestCircle) {
        let [x, y, r] = bestCircle;
        let mask = cv.Mat.zeros(eyeRegion.rows, eyeRegion.cols, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(x, y), r, [255, 255, 255, 255], -1);
        let finalIntensity = cv.mean(eyeRegion, mask)[0];
        mask.delete();
        
        addLog(`\n✅ TERPILIH [${bestStrategy}]:`, 'success');
        addLog(`   Position: (${x}, ${y})`, 'info');
        addLog(`   Radius: ${r} px`, 'info');
        addLog(`   Intensity: ${finalIntensity.toFixed(1)}`, 'info');
        addLog(`   Score: ${bestScore.toFixed(3)}`, 'info');
        
        if (finalIntensity > 80) {
            addLog(`   ⚠️  WARNING: Intensity tinggi! Mungkin bukan pupil.`, 'warning');
        }
    } else {
        addLog(`\n❌ Tidak ada pupil terdeteksi`, 'error');
        addLog(`   Tip: Coba enable debug=True untuk lihat preprocessing`, 'info');
    }
    
    return bestCircle;
}

function detectPupilWithFallback(eyeRegion, eyeName = "Eye") {
    let circle = detectPupilImproved(eyeRegion, eyeName, false);
    
    if (circle) {
        // Check confidence based on intensity
        let mask = cv.Mat.zeros(eyeRegion.rows, eyeRegion.cols, cv.CV_8UC1);
        cv.circle(mask, new cv.Point(circle[0], circle[1]), circle[2], [255, 255, 255, 255], -1);
        let intensity = cv.mean(eyeRegion, mask)[0];
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

function visualizeResult(eyeRegion, circle, eyeName, confidence, canvasId) {
    if (!eyeRegion) return;
    
    let output = new cv.Mat();
    cv.cvtColor(eyeRegion, output, cv.COLOR_GRAY2RGBA);
    
    if (circle) {
        let [x, y, r] = circle;
        
        // Warna berdasarkan confidence
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
        
        // Add label
        if (confidence) {
            cv.putText(output, `Conf: ${confidence}`, new cv.Point(5, 15),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255, 255], 1);
        }
    }
    
    cv.imshow(canvasId, output);
    output.delete();
}

// ================================================================================
// CALCULATE PUPIL POSITION DIFFERENCE (sama persis dengan Python)
// ================================================================================
function calculatePupilPositionDifference(pupils, eyes) {
    if (!pupils[0] || !pupils[1]) {
        addLog("Pupil tidak terdeteksi pada salah satu atau kedua mata", 'error');
        return null;
    }
    
    // Extract pupil x positions
    let leftPupilX = pupils[0][0];
    let rightPupilX = pupils[1][0];
    
    // Extract eye widths
    let leftEyeW = eyes[0].width;
    let rightEyeW = eyes[1].width;
    
    // Normalize pupil positions within their respective eye bounding boxes
    let leftNormalized = leftPupilX / leftEyeW;   // 0 = kiri, 1 = kanan
    let rightNormalized = rightPupilX / rightEyeW; // 0 = kiri, 1 = kanan
    
    // Hitung perbedaan posisi antara kedua pupil
    let dx = leftNormalized - rightNormalized;
    
    return {
        dx: dx,
        leftNorm: leftNormalized,
        rightNorm: rightNormalized
    };
}

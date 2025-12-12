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

// ===== PYTHON: def detect_face(gray) =====
function detect_face(gray) {
    console.log("Starting face detection...");
    console.log("Gray image type:", gray.type(), "Size:", gray.cols, "x", gray.rows);
    
    // faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    let faces = new cv.RectVector();
    let minSize = new cv.Size(80, 80);
    
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, minSize);

    console.log("Faces detected:", faces.size());

    // if len(faces) == 0: return None, None
    if (faces.size() === 0) {
        console.log("❌ Tidak ada wajah terdeteksi");
        
        // Try with more lenient parameters
        console.log("Trying with more lenient parameters...");
        faceCascade.detectMultiScale(gray, faces, 1.05, 3, 0, new cv.Size(50, 50));
        console.log("Faces detected with lenient params:", faces.size());
        
        if (faces.size() === 0) {
            faces.delete();
            return [null, null];
        }
    }

    // faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
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
    
    // (fx, fy, fw, fh) = faces_sorted[0]
    let largestFace = facesList[0];
    let fx = largestFace.x, fy = largestFace.y, fw = largestFace.width, fh = largestFace.height;

    // face_gray = gray[fy:fy+fh, fx:fx+fw]
    let face_gray = gray.roi(new cv.Rect(fx, fy, fw, fh));

    faces.delete();

    // return (fx, fy, fw, fh), face_gray
    return [[fx, fy, fw, fh], face_gray];
}

// ===== PYTHON: def improved_eye_detection(gray) =====
function improved_eye_detection(gray) {
    // scaleFactors = [1.01, 1.02, 1.05, 1.1]
    const scaleFactors = [1.01, 1.02, 1.05, 1.1];
    // minNeighbors_list = [1, 2, 3]
    const minNeighbors_list = [1, 2, 3];
    // minSizes = [(10, 10), (15, 15), (20, 20), (25, 25)]
    const minSizes = [
        new cv.Size(10, 10),
        new cv.Size(15, 15),
        new cv.Size(20, 20),
        new cv.Size(25, 25)
    ];

    let best_eyes = null;
    let best_score = -1;
    let best_params = null;

    // for sf in scaleFactors:
    for (let sf of scaleFactors) {
        // for mn in minNeighbors_list:
        for (let mn of minNeighbors_list) {
            // for ms in minSizes:
            for (let ms of minSizes) {
                let eyes = new cv.RectVector();
                
                try {
                    // eyes = eye_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=ms, flags=cv2.CASCADE_SCALE_IMAGE)
                    eyeCascade.detectMultiScale(gray, eyes, sf, mn, 0, ms);

                    // if len(eyes) >= 2:
                    if (eyes.size() >= 2) {
                        // areas = [(ex, ey, ew, eh, ew*eh) for (ex, ey, ew, eh) in eyes]
                        let areas = [];
                        for (let i = 0; i < eyes.size(); i++) {
                            let eye = eyes.get(i);
                            areas.push([eye.x, eye.y, eye.width, eye.height, eye.width * eye.height]);
                        }

                        // areas_sorted = sorted(areas, key=lambda x: x[4], reverse=True)[:2]
                        areas.sort((a, b) => b[4] - a[4]);
                        let areas_sorted = areas.slice(0, 2);

                        // eye1, eye2 = areas_sorted[0][:4], areas_sorted[1][:4]
                        let eye1 = areas_sorted[0].slice(0, 4);
                        let eye2 = areas_sorted[1].slice(0, 4);
                        
                        // x1, y1, w1, h1 = eye1
                        let x1 = eye1[0], y1 = eye1[1], w1 = eye1[2], h1 = eye1[3];
                        // x2, y2, w2, h2 = eye2
                        let x2 = eye2[0], y2 = eye2[1], w2 = eye2[2], h2 = eye2[3];

                        // dx = min(x1+w1, x2+w2) - max(x1, x2)
                        let dx = Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2);
                        // dy = min(y1+h1, y2+h2) - max(y1, y2)
                        let dy = Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2);

                        // if dx <= 0 or dy <= 0:
                        if (dx <= 0 || dy <= 0) {
                            // score = sum([a[4] for a in areas_sorted])
                            let score = areas_sorted[0][4] + areas_sorted[1][4];
                            
                            // if score > best_score:
                            if (score > best_score) {
                                best_score = score;
                                // best_eyes = [(int(eye1[0]), int(eye1[1]), int(eye1[2]), int(eye1[3])), ...]
                                best_eyes = [
                                    [Math.round(x1), Math.round(y1), Math.round(w1), Math.round(h1)],
                                    [Math.round(x2), Math.round(y2), Math.round(w2), Math.round(h2)]
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

    // return (best_eyes if best_eyes else [], best_params)
    return [best_eyes ? best_eyes : [], best_params];
}

// ===== PYTHON: def detect_pupil_hough_circle_auto_simple(eye_gray, eye_title="Eye") =====
function detect_pupil_hough_circle_auto_simple(eye_gray, eye_title = "Eye") {
    // if eye_gray is None or eye_gray.size == 0: return None
    if (!eye_gray || eye_gray.rows === 0 || eye_gray.cols === 0) {
        console.log(`\n${eye_title} — NO circles detected!`);
        return null;
    }

    // eye_height, eye_width = eye_gray.shape
    let eye_height = eye_gray.rows;
    let eye_width = eye_gray.cols;

    // blurred = cv2.GaussianBlur(eye_gray, (7,7), 2)
    let blurred = new cv.Mat();
    cv.GaussianBlur(eye_gray, blurred, new cv.Size(7, 7), 2, 2, cv.BORDER_DEFAULT);

    // param_sets = [...]
    const param_sets = [
        {dp: 1.1, param1: 80, param2: 20, minRadius: 15, maxRadius: 40},
        {dp: 1.0, param1: 60, param2: 15, minRadius: 10, maxRadius: 45},
        {dp: 0.9, param1: 40, param2: 10, minRadius: 5, maxRadius: 50},
        {dp: 1.2, param1: 100, param2: 25, minRadius: 20, maxRadius: 35},
        {dp: 1.0, param1: 90, param2: 30, minRadius: 8, maxRadius: 30}
    ];

    let all_circles = [];

    // for params in param_sets:
    for (let params of param_sets) {
        try {
            let circles = new cv.Mat();
            // circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=params['dp'], minDist=20, ...)
            cv.HoughCircles(blurred, circles, cv.HOUGH_GRADIENT, params.dp, 20, params.param1, params.param2, params.minRadius, params.maxRadius);

            // if circles is not None:
            if (circles.cols > 0) {
                // circles_rounded = np.round(circles[0, :]).astype("int")
                // for circle in circles_rounded:
                for (let i = 0; i < circles.cols; i++) {
                    let x = Math.round(circles.data32F[i * 3]);
                    let y = Math.round(circles.data32F[i * 3 + 1]);
                    let r = Math.round(circles.data32F[i * 3 + 2]);

                    // if (x - r > 0 and x + r < eye_width and y - r > 0 and y + r < eye_height and 5 <= r <= 50):
                    if (x - r > 0 && x + r < eye_width && y - r > 0 && y + r < eye_height && r >= 5 && r <= 50) {
                        all_circles.push([x, y, r]);
                    }
                }
            }
            circles.delete();
        } catch(e) {
            continue;
        }
    }

    blurred.delete();

    let best_circle = null;
    
    // if len(all_circles) > 0:
    if (all_circles.length > 0) {
        // center_x, center_y = eye_width // 2, eye_height // 2
        let center_x = Math.floor(eye_width / 2);
        let center_y = Math.floor(eye_height / 2);

        let scored_circles = [];
        
        // for circle in all_circles:
        for (let circle of all_circles) {
            let x = circle[0], y = circle[1], r = circle[2];
            
            // distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            let distance = Math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2);

            // center_score = 1.0 / (1.0 + distance / max(eye_width, eye_height))
            let center_score = 1.0 / (1.0 + distance / Math.max(eye_width, eye_height));

            // Radius score: prefer 15-35 pixels
            let radius_score;
            if (r >= 15 && r <= 35) {
                radius_score = 1.0;
            } else if (r >= 10 && r <= 40) {
                radius_score = 0.7;
            } else {
                radius_score = 0.3;
            }

            // total_score = center_score * 0.7 + radius_score * 0.3
            let total_score = center_score * 0.7 + radius_score * 0.3;
            scored_circles.push([total_score, circle]);
        }

        // scored_circles.sort(reverse=True, key=lambda x: x[0])
        scored_circles.sort((a, b) => b[0] - a[0]);
        // best_circle = scored_circles[0][1]
        best_circle = scored_circles[0][1];
    }

    // if best_circle is not None: print(...) (x, y, r) = best_circle
    if (best_circle !== null) {
        console.log(`\n${eye_title} — Circles detected:`);
        console.log(best_circle);
    } else {
        console.log(`\n${eye_title} — NO circles detected!`);
        return null;
    }

    // return np.array([best_circle]) if best_circle is not None else None
    return best_circle ? [best_circle] : null;
}

// ===== PYTHON: def calculate_pupil_position_difference(pupils, eyes) =====
function calculate_pupil_position_difference(pupils, eyes) {
    // if pupils[0][0] is None or pupils[1][0] is None: return None
    if (pupils[0][0] === null || pupils[1][0] === null) {
        console.log("Pupil tidak terdeteksi pada salah satu atau kedua mata");
        return null;
    }

    // left_pupil_x, left_pupil_y, left_pupil_r = pupils[0]
    let left_pupil_x = pupils[0][0], left_pupil_y = pupils[0][1], left_pupil_r = pupils[0][2];
    // right_pupil_x, right_pupil_y, right_pupil_r = pupils[1]
    let right_pupil_x = pupils[1][0], right_pupil_y = pupils[1][1], right_pupil_r = pupils[1][2];

    // left_eye_x, left_eye_y, left_eye_w, left_eye_h = eyes[0]
    let left_eye_x = eyes[0][0], left_eye_y = eyes[0][1], left_eye_w = eyes[0][2], left_eye_h = eyes[0][3];
    // right_eye_x, right_eye_y, right_eye_w, right_eye_h = eyes[1]
    let right_eye_x = eyes[1][0], right_eye_y = eyes[1][1], right_eye_w = eyes[1][2], right_eye_h = eyes[1][3];

    // left_normalized = left_pupil_x / left_eye_w
    let left_normalized = left_pupil_x / left_eye_w;
    // right_normalized = right_pupil_x / right_eye_w
    let right_normalized = right_pupil_x / right_eye_w;

    // dx = left_normalized - right_normalized
    let dx = left_normalized - right_normalized;

    // return dx, left_normalized, right_normalized
    return [dx, left_normalized, right_normalized];
}

// ===== MAIN ANALYSIS (SAMA SEPERTI PYTHON FLOW) =====
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
    // img = image.load_img(filename); gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    let img = filePreview;
    let src = cv.imread(img);
    
    console.log("Image loaded - Size:", src.cols, "x", src.rows, "Channels:", src.channels());
    
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    
    console.log("Gray image - Size:", gray.cols, "x", gray.rows);

    // face_box, face_gray = detect_face(gray)
    let [face_box, face_gray] = detect_face(gray);

    // if face_gray is None:
    if (face_gray === null) {
        showResult("❌ Gagal lanjut ke eye detection karena wajah tidak ditemukan.");
        cleanup(src, gray);
        return;
    }

    console.log("Face detected:", face_box);

    // eyes_in_face, params = improved_eye_detection(face_gray)
    let [eyes_in_face, params] = improved_eye_detection(face_gray);

    console.log("Eyes dalam face:", eyes_in_face);
    console.log("Best params:", params);

    // eyes = sorted(eyes_in_face, key=lambda e: e[0])
    let eyes = eyes_in_face.sort((a, b) => a[0] - b[0]);
    console.log("Final eyes (sorted):", eyes);

    if (eyes.length < 2) {
        showResult("❌ Mata tidak ditemukan.");
        cleanup(src, gray, face_gray);
        return;
    }

    // (ex, ey, ew, eh) = eyes[0]; left_eye = face_gray[ey:ey+eh, ex:ex+ew]
    let ex = eyes[0][0], ey = eyes[0][1], ew = eyes[0][2], eh = eyes[0][3];
    let left_eye = face_gray.roi(new cv.Rect(ex, ey, ew, eh));

    // (ex, ey, ew, eh) = eyes[1]; right_eye = face_gray[ey:ey+eh, ex:ex+ew]
    ex = eyes[1][0]; ey = eyes[1][1]; ew = eyes[1][2]; eh = eyes[1][3];
    let right_eye = face_gray.roi(new cv.Rect(ex, ey, ew, eh));

    // left_circles = detect_pupil_hough_circle_auto_simple(left_eye, "Left Eye")
    let left_circles = detect_pupil_hough_circle_auto_simple(left_eye, "Left Eye");
    // right_circles = detect_pupil_hough_circle_auto_simple(right_eye, "Right Eye")
    let right_circles = detect_pupil_hough_circle_auto_simple(right_eye, "Right Eye");

    // pupils = []
    let pupils = [];

    // if left_circles is not None: pupils.append(tuple(left_circles[0]))
    if (left_circles !== null) {
        pupils.push(left_circles[0]);
    } else {
        pupils.push([null, null, null]);
    }

    // if right_circles is not None: pupils.append(tuple(right_circles[0]))
    if (right_circles !== null) {
        pupils.push(right_circles[0]);
    } else {
        pupils.push([null, null, null]);
    }

    // result = calculate_pupil_position_difference(pupils, eyes)
    let result = calculate_pupil_position_difference(pupils, eyes);

    console.log("\n========= PUPIL POSITION DIFFERENCE ANALYSIS =========");

    if (result !== null) {
        // dx, left_norm, right_norm = result
        let dx = result[0], left_norm = result[1], right_norm = result[2];

        console.log(`Left Eye Normalized X:  ${left_norm.toFixed(3)}`);
        console.log(`Right Eye Normalized X: ${right_norm.toFixed(3)}`);
        console.log(`Difference (dx):        ${dx.toFixed(3)}`);
        console.log(`Average Position:       ${((left_norm + right_norm) / 2).toFixed(3)}`);
        console.log(`\nNilai dx untuk gambar ini: ${dx.toFixed(6)}`);

        // Draw visualizations
        drawEyeVisualization(left_eye, pupils[0], "leftEyeCanvas", "Left Eye");
        drawEyeVisualization(right_eye, pupils[1], "rightEyeCanvas", "Right Eye");

        let htmlOutput = `
            <div class="result-item"><b>Left Pupil:</b> (x: ${pupils[0][0]}, y: ${pupils[0][1]}, r: ${pupils[0][2]})</div>
            <div class="result-item"><b>Right Pupil:</b> (x: ${pupils[1][0]}, y: ${pupils[1][1]}, r: ${pupils[1][2]})</div>
            <div class="result-item"><b>Left Eye Normalized X:</b> ${left_norm.toFixed(3)}</div>
            <div class="result-item"><b>Right Eye Normalized X:</b> ${right_norm.toFixed(3)}</div>
            <div class="result-item"><b>Difference (dx):</b> ${dx.toFixed(3)}</div>
            <div class="result-item"><b>Average Position:</b> ${((left_norm + right_norm) / 2).toFixed(3)}</div>
            <div class="result-item" style="margin-top: 15px; font-style: italic;">
                Nilai dx untuk gambar ini: ${dx.toFixed(6)}
            </div>
        `;
        showResult(htmlOutput);
    } else {
        console.log("Tidak dapat menghitung perbedaan posisi pupil");
        showResult("❌ Tidak dapat menghitung perbedaan posisi pupil");
    }

    cleanup(src, gray, face_gray, left_eye, right_eye);
}

function drawEyeVisualization(eye_gray, pupil, canvasId, eye_title) {
    // output = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2RGB)
    let output = new cv.Mat();
    cv.cvtColor(eye_gray, output, cv.COLOR_GRAY2RGB);

    if (pupil[0] !== null) {
        // (x, y, r) = best_circle
        let x = pupil[0], y = pupil[1], r = pupil[2];
        
        // cv2.circle(output, (x, y), r, (0,255,0), 2)
        cv.circle(output, new cv.Point(x, y), r, [0, 255, 0, 255], 2);
        // cv2.circle(output, (x, y), 2, (255,0,0), 3)
        cv.circle(output, new cv.Point(x, y), 2, [255, 0, 0, 255], 3);
    }

    // cv.imshow(canvas, output)
    let canvas = document.getElementById(canvasId);
    cv.imshow(canvas, output);

    output.delete();
}

function showResult(text) {
    loadingDiv.style.display = "none";
    resultBox.style.display = "block";
    resultContent.innerHTML = text;
}

function cleanup(...mats) {
    mats.forEach(m => m && m.delete());
}

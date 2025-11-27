// Element references
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const filePreview = document.getElementById('filePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resultContent = document.getElementById('resultContent');

// Event listener untuk file input
fileInput.addEventListener('change', function(e) {
  const file = e.target.files[0];
  
  if (file) {
    // Validasi file
    if (!file.type.startsWith('image/')) {
      alert('Harap pilih file gambar!');
      return;
    }
    
    // Tampilkan info file
    fileInfo.textContent = `File: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
    
    // Tampilkan preview gambar
    const reader = new FileReader();
    reader.onload = function(e) {
      filePreview.src = e.target.result;
      filePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Aktifkan tombol analisis
    analyzeBtn.disabled = false;
  }
});

// Event listener untuk tombol analisis
analyzeBtn.addEventListener('click', async function() {
  const file = fileInput.files[0];
  
  if (!file) {
    alert('Harap pilih file terlebih dahulu!');
    return;
  }
  
  // Tampilkan loading
  loading.style.display = 'block';
  analyzeBtn.disabled = true;
  results.style.display = 'none';
  
  try {
    // Kirim file ke server
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Server error: ' + response.status);
    }
    
    const result = await response.json();
    
    // Tampilkan hasil
    displayResults(result);
    
  } catch (error) {
    console.error('Error:', error);
    alert('Terjadi kesalahan saat menganalisis gambar: ' + error.message);
  } finally {
    // Sembunyikan loading
    loading.style.display = 'none';
    analyzeBtn.disabled = false;
  }
});

// Fungsi untuk menampilkan hasil
function displayResults(data) {
  let html = '';
  
  if (data.success) {
    html = `
      <div class="result-item">
        <strong>Status:</strong> ✅ Analisis Berhasil
      </div>
      <div class="result-item">
        <strong>File:</strong> ${data.filename}
      </div>
      <div class="result-item">
        <strong>Mata Kanan:</strong> ${data.right_eye || 'Tidak terdeteksi'}
      </div>
      <div class="result-item">
        <strong>Mata Kiri:</strong> ${data.left_eye || 'Tidak terdeteksi'}
      </div>
      <div class="result-item">
        <strong>DX Value:</strong> ${data.dx || 'Tidak dapat dihitung'}
      </div>
      <div class="result-item">
        <strong>Keterangan:</strong> ${data.interpretation || ''}
      </div>
    `;
    
    // Tambahkan interpretasi medis sederhana
    if (data.dx && !isNaN(data.dx)) {
      const dxValue = parseFloat(data.dx);
      let interpretation = '';
      
      if (dxValue < 0.05) {
        interpretation = 'Normal - Posisi pupil simetris';
      } else if (dxValue < 0.1) {
        interpretation = 'Borderline - Perbedaan posisi pupil kecil';
      } else {
        interpretation = 'Abnormal - Kemungkinan strabismus';
      }
      
      html += `<div class="result-item" style="background: #fff3cd;">
        <strong>Interpretasi:</strong> ${interpretation}
      </div>`;
    }
    
  } else {
    html = `
      <div class="result-item" style="background: #f8d7da;">
        <strong>Status:</strong> ❌ Analisis Gagal
      </div>
      <div class="result-item">
        <strong>Error:</strong> ${data.error || 'Unknown error'}
      </div>
    `;
  }
  
  resultContent.innerHTML = html;
  results.style.display = 'block';
  
  // Scroll ke hasil
  results.scrollIntoView({ behavior: 'smooth' });
}

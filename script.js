(function(){
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const canvasWrap = document.getElementById('canvasWrap');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const veil = document.getElementById('veil');
  const veilText = document.getElementById('veilText');
  const removeBtn = document.getElementById('removeBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const resetBtn = document.getElementById('resetBtn');
  const postActions = document.getElementById('postActions');
  const editControls = document.getElementById('editControls');
  const startBlock = document.getElementById('startBlock');
  const canvasToolbar = document.getElementById('canvasToolbar');
  const brushHint = document.getElementById('brushHint');
  const brushSize = document.getElementById('brushSize');
  const brushSizeLabel = document.getElementById('brushSizeLabel');
  const smoothing = document.getElementById('smoothing');
  const smoothLabel = document.getElementById('smoothLabel');
  const swatches = document.getElementById('swatches');
  const galleryGrid = document.getElementById('galleryGrid');
  const toast = document.getElementById('toast');

  let originalImageData = null; // pristine pixel data at load
  let workingImageData = null;  // current pixels (with alpha edits)
  let bgRemoved = false;
  let mode = 'auto';
  let selectedBg = 'transparent';
  let painting = false;

  // Classes the system actively look for when detecting an object. If neither is found,
  // system fall back to whatever other objects the model did detect, then to a
  // and plain color-based cutout as a last resort.
  const PRIORITY_CLASSES = ['person', 'car'];
  let deeplabModel = null;
  let modelLoadPromise = null;

  function loadDetectionModel(){
    if(modelLoadPromise) return modelLoadPromise;
    if(typeof deeplab === 'undefined'){
      return Promise.reject(new Error('Detection library did not load'));
    }
    modelLoadPromise = deeplab.load({ base: 'pascal', quantizationBytes: 2 })
      .then(m => { deeplabModel = m; return m; });
    return modelLoadPromise;
  }

  function showToast(msg){
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(()=> toast.classList.remove('show'), 2600);
  }
  window.showToast = showToast;

  //  Upload handling 
  dropzone.addEventListener('click', ()=> fileInput.click());
  ['dragover','dragenter'].forEach(ev=>{
    dropzone.addEventListener(ev, e=>{ e.preventDefault(); dropzone.style.background = 'rgba(108,76,241,0.06)'; });
  });
  ['dragleave','drop'].forEach(ev=>{
    dropzone.addEventListener(ev, e=>{ e.preventDefault(); dropzone.style.background = 'transparent'; });
  });
  dropzone.addEventListener('drop', e=>{
    const f = e.dataTransfer.files[0];
    if(f) loadFile(f);
  });
  fileInput.addEventListener('change', e=>{
    if(e.target.files[0]) loadFile(e.target.files[0]);
  });

  function loadFile(file){
    if(!file.type.startsWith('image/')){ showToast('Please choose an image file.'); return; }
    const reader = new FileReader();
    reader.onload = e=>{
      const img = new Image();
      img.onload = ()=> setupCanvas(img);
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function setupCanvas(img){
    const maxDim = 720;
    let w = img.width, h = img.height;
    const scale = Math.min(1, maxDim / Math.max(w,h));
    w = Math.round(w*scale); h = Math.round(h*scale);
    canvas.width = w; canvas.height = h;
    ctx.clearRect(0,0,w,h);
    ctx.drawImage(img, 0, 0, w, h);
    originalImageData = ctx.getImageData(0,0,w,h);
    workingImageData = ctx.getImageData(0,0,w,h);
    bgRemoved = false;

    dropzone.style.display = 'none';
    canvas.style.display = 'block';
    canvasWrap.classList.add('has-image');
    canvasWrap.classList.remove('checker');
    startBlock.style.display = 'none';
    removeBtn.disabled = false;
    postActions.style.display = 'none';
    editControls.style.display = 'none';
    canvasToolbar.style.display = 'none';
    brushHint.style.display = 'none';
    removeBtn.textContent = 'Remove Background';
  }

  //  used TensorFlow.js DeepLab for background removal
  removeBtn.addEventListener('click', async ()=>{
    if(bgRemoved) return;
    veil.style.display = 'flex';
    removeBtn.disabled = true;
    let detectedLabel = null;

    try{
      veilText.textContent = 'Loading detection model…';
      await loadDetectionModel();
      veilText.textContent = 'Looking for a person or car…';
      detectedLabel = await runObjectSegmentation();
    } catch(err){
      console.warn('Object detection unavailable, using color-based cutout instead:', err);
      veilText.textContent = 'Using a quick edge-based cutout…';
      await new Promise(r=>setTimeout(r, 350));
      runColorRemoval();
    }

    veil.style.display = 'none';
    bgRemoved = true;
    postActions.style.display = 'flex';
    editControls.style.display = 'block';
    canvasToolbar.style.display = 'flex';
    canvasWrap.classList.add('checker');
    brushHint.style.display = 'block';
    removeBtn.textContent = 'Background removed';
    addToGallery();
    showToast(detectedLabel ? `Found a ${detectedLabel} — background removed` : 'Background removed');
  });

  // Runs semantic segmentation (TensorFlow.js DeepLab, Pascal VOC classes)
  // and keeps only the pixels belonging to a detected person/car (or, failing
  // that, whatever object the model did find). Returns the label it kept, or
  // throws if nothing usable was detected so the caller can fall back.
  async function runObjectSegmentation(){
    const w = canvas.width, h = canvas.height;

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = w; srcCanvas.height = h;
    srcCanvas.getContext('2d').putImageData(originalImageData, 0, 0);

    const result = await deeplabModel.segment(srcCanvas);
    const { legend, height: mh, width: mw, segmentationMap } = result;

    let keepNames = PRIORITY_CLASSES.filter(name => legend[name]);
    if(keepNames.length === 0){
      keepNames = Object.keys(legend).filter(name => name !== 'background');
    }
    if(keepNames.length === 0){
      throw new Error('No recognizable object found in this photo');
    }
    const keepColors = keepNames.map(name => legend[name]);

    // Build a mask at the model's native resolution, then scale it up.
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = mw; maskCanvas.height = mh;
    const maskCtx = maskCanvas.getContext('2d');
    const maskData = maskCtx.createImageData(mw, mh);
    for(let i = 0; i < segmentationMap.length; i += 4){
      const r = segmentationMap[i], g = segmentationMap[i+1], b = segmentationMap[i+2];
      const isKept = keepColors.some(c => c[0] === r && c[1] === g && c[2] === b);
      maskData.data[i]   = 255;
      maskData.data[i+1] = 255;
      maskData.data[i+2] = 255;
      maskData.data[i+3] = isKept ? 255 : 0;
    }
    maskCtx.putImageData(maskData, 0, 0);

    const scaledMask = document.createElement('canvas');
    scaledMask.width = w; scaledMask.height = h;
    const smCtx = scaledMask.getContext('2d');
    smCtx.imageSmoothingEnabled = true;
    smCtx.drawImage(maskCanvas, 0, 0, w, h);
    const maskPixels = smCtx.getImageData(0, 0, w, h).data;

    const src = originalImageData.data;
    const out = ctx.createImageData(w, h);
    for(let i = 0; i < src.length; i += 4){
      out.data[i]   = src[i];
      out.data[i+1] = src[i+1];
      out.data[i+2] = src[i+2];
      out.data[i+3] = maskPixels[i+3] > 128 ? 255 : 0;
    }
    workingImageData = out;
    applyBackground();
    return keepNames.join(' + ');
  }

  // Fallback used only if the detection model can't load (offline, blocked
  // CDN, etc.) — a plain color-similarity cutout based on the corner colors.
  function runColorRemoval(){
    const w = canvas.width, h = canvas.height;
    const src = originalImageData.data;
    const out = ctx.createImageData(w,h);
    const data = out.data;

    const corners = [
      [1,1], [w-2,1], [1,h-2], [w-2,h-2]
    ];
    let br=0,bg=0,bb=0;
    corners.forEach(([x,y])=>{
      const i = (y*w+x)*4;
      br += src[i]; bg += src[i+1]; bb += src[i+2];
    });
    br/=4; bg/=4; bb/=4;

    const tolerance = 42 + (100 - parseInt(smoothing.value,10)) * 0.3;

    for(let i=0;i<src.length;i+=4){
      const r=src[i], g=src[i+1], b=src[i+2];
      const dist = Math.sqrt((r-br)**2 + (g-bg)**2 + (b-bb)**2);
      data[i]=r; data[i+1]=g; data[i+2]=b;
      data[i+3] = dist < tolerance ? 0 : 255;
    }
    ctx.putImageData(out,0,0);
    workingImageData = ctx.getImageData(0,0,w,h);
    applyBackground();
  }

  function applyBackground(){
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    if(selectedBg === 'transparent'){
      ctx.putImageData(workingImageData, 0, 0);
      return;
    }
    // Composite: paint the color/gradient first, then draw the cutout (with alpha) on top
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(workingImageData, 0, 0);
    if(selectedBg === 'gradient'){
      const g = ctx.createLinearGradient(0,0,w,h);
      g.addColorStop(0,'#6c4cf1'); g.addColorStop(1,'#241d55');
      ctx.fillStyle = g; ctx.fillRect(0,0,w,h);
    } else {
      ctx.fillStyle = selectedBg; ctx.fillRect(0,0,w,h);
    }
    ctx.drawImage(tmp,0,0);
  }

  //  Quick background swatches 
  swatches.addEventListener('click', e=>{
    const sw = e.target.closest('.swatch');
    if(!sw) return;
    [...swatches.children].forEach(c=>c.classList.remove('selected'));
    sw.classList.add('selected');
    selectedBg = sw.dataset.bg;
    if(bgRemoved) applyBackground();
  });

  //  Erase / Restore brush 
  canvasToolbar.addEventListener('click', e=>{
    const btn = e.target.closest('button');
    if(!btn) return;
    mode = btn.dataset.mode;
    [...canvasToolbar.children].forEach(c=>c.classList.remove('active'));
    btn.classList.add('active');
  });

  function canvasPos(e){
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const point = (e.touches ? e.touches[0] : e);
    return { x: (point.clientX - rect.left)*scaleX, y: (point.clientY - rect.top)*scaleY };
  }

  function paintAt(x,y){
    if(mode === 'auto' || !bgRemoved) return;
    const r = parseInt(brushSize.value,10)/2;
    const w = canvas.width, h = canvas.height;
    const data = workingImageData.data;
    const x0 = Math.max(0, Math.floor(x-r)), x1 = Math.min(w, Math.ceil(x+r));
    const y0 = Math.max(0, Math.floor(y-r)), y1 = Math.min(h, Math.ceil(y+r));
    for(let yy=y0; yy<y1; yy++){
      for(let xx=x0; xx<x1; xx++){
        if((xx-x)**2 + (yy-y)**2 <= r*r){
          const i = (yy*w+xx)*4;
          if(mode === 'erase'){
            data[i+3] = 0;
          } else if(mode === 'restore'){
            data[i]   = originalImageData.data[i];
            data[i+1] = originalImageData.data[i+1];
            data[i+2] = originalImageData.data[i+2];
            data[i+3] = 255;
          }
        }
      }
    }
    applyBackground();
  }

  canvas.addEventListener('mousedown', e=>{ painting = true; const p = canvasPos(e); paintAt(p.x,p.y); });
  canvas.addEventListener('mousemove', e=>{ if(painting){ const p = canvasPos(e); paintAt(p.x,p.y); } });
  window.addEventListener('mouseup', ()=> painting=false);
  canvas.addEventListener('touchstart', e=>{ painting=true; const p=canvasPos(e); paintAt(p.x,p.y); e.preventDefault(); }, {passive:false});
  canvas.addEventListener('touchmove', e=>{ if(painting){ const p=canvasPos(e); paintAt(p.x,p.y); } e.preventDefault(); }, {passive:false});
  canvas.addEventListener('touchend', ()=> painting=false);

  brushSize.addEventListener('input', ()=> brushSizeLabel.textContent = brushSize.value+'px');
  smoothing.addEventListener('input', ()=> smoothLabel.textContent = smoothing.value+'%');

  //  Download 
  downloadBtn.addEventListener('click', ()=>{
    const link = document.createElement('a');
    link.download = 'cleansnap-cutout.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
    showToast('Downloaded PNG');
  });

  //  Reset 
  resetBtn.addEventListener('click', ()=>{
    fileInput.value = '';
    canvas.style.display = 'none';
    dropzone.style.display = 'block';
    canvasWrap.classList.remove('has-image');
    canvasWrap.classList.add('checker');
    startBlock.style.display = 'block';
    removeBtn.disabled = true;
    postActions.style.display = 'none';
    editControls.style.display = 'none';
    canvasToolbar.style.display = 'none';
    brushHint.style.display = 'none';
    bgRemoved = false;
  });

  //  Gallery 
  function addToGallery(){
    const empty = galleryGrid.querySelector('.gallery-empty');
    if(empty) empty.remove();
    const item = document.createElement('div');
    item.className = 'gallery-item checker';
    const img = document.createElement('img');
    img.src = canvas.toDataURL('image/png');
    item.appendChild(img);
    galleryGrid.prepend(item);
  }

  //  Tool tabs (Batch / Templates are demo-only) 
  document.querySelectorAll('.tool-tab').forEach(tab=>{
    tab.addEventListener('click', ()=>{
      document.querySelectorAll('.tool-tab').forEach(t=>t.classList.remove('active'));
      tab.classList.add('active');
      if(tab.dataset.tab !== 'upload'){
        showToast(tab.textContent + ' is part of the Pro plan.');
        tab.classList.remove('active');
        document.querySelector('.tool-tab[data-tab="upload"]').classList.add('active');
      }
    });
  });
})();

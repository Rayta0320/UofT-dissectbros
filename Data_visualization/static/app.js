const state={subjects:[],orientation:'axial',alignment:'peak',position:500,mode:'overlay',opacity:62,outline:false,center:40,window:400,generation:0,playing:false};
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const grid=$('#grid');

function axisIndex(subject){const axis={sagittal:0,coronal:1,axial:2}[state.orientation];if(state.alignment==='peak'&&subject.peak)return subject.peak[axis];return Math.round(state.position/1000*(subject.shape[axis]-1));}
function urlFor(subject){const q=new URLSearchParams({orientation:state.orientation,alignment:state.alignment,position:state.position/1000,mode:state.mode,opacity:state.opacity/100,outline:state.outline?'1':'0',center:state.center,window:state.window,t:state.generation});return `/api/subjects/${subject.id}/slice?${q}`;}
function makeCards(){grid.innerHTML='';for(const subject of state.subjects){const card=$('#card-template').content.firstElementChild.cloneNode(true);card.dataset.id=subject.id;card.querySelector('strong').textContent=`Subject ${String(subject.id).padStart(3,'0')}`;card.querySelector('.dimensions').textContent=subject.shape.join(' × ');card.querySelector('.spacing').textContent=subject.labeled_voxels?`${subject.labeled_voxels.toLocaleString()} labeled voxels`:subject.spacing.join(' × ')+' mm';card.querySelector('img').alt=`Subject ${subject.id} scan and segmentation`;grid.append(card)}filterCards();}
function updateMeta(){for(const subject of state.subjects){const card=grid.querySelector(`[data-id="${subject.id}"]`);const axis={sagittal:0,coronal:1,axial:2}[state.orientation];card.querySelector('.slice-index').textContent=`${state.orientation[0].toUpperCase()} · ${axisIndex(subject)+1}/${subject.shape[axis]}`;card.classList.toggle('split',state.mode==='split')}}

let queue=[],active=0;
function renderAll(){state.generation++;queue=[];updateMeta();const generation=state.generation;for(const subject of state.subjects){const card=grid.querySelector(`[data-id="${subject.id}"]`);if(card.hidden)continue;const img=card.querySelector('img');img.classList.remove('loaded');card.querySelector('.viewport').classList.remove('failed');card.querySelector('.loader').style.display='block';queue.push({subject,card,img,generation})}pump();}
function pump(){while(active<4&&queue.length){const job=queue.shift();active++;const preload=new Image();preload.onload=()=>finish(job,preload.src,true);preload.onerror=()=>finish(job,'',false);preload.src=urlFor(job.subject)}if(active===0&&queue.length===0){$('#status').textContent=`${state.subjects.length} subjects ready`;$('#status-dot').classList.add('ready')}}
function finish(job,src,ok){active--;if(job.generation===state.generation){job.card.querySelector('.loader').style.display='none';if(ok){job.img.src=src;job.img.classList.add('loaded')}else job.card.querySelector('.viewport').classList.add('failed')}pump();}
let debounce;
function schedule(){clearTimeout(debounce);debounce=setTimeout(renderAll,90)}
function activate(container,value){container.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.value===value));}
$('#orientation').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;state.orientation=b.dataset.value;activate($('#orientation'),state.orientation);renderAll()});
$('#alignment').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;state.alignment=b.dataset.value;activate($('#alignment'),state.alignment);setAlignmentControls();renderAll()});
$('#mode').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;state.mode=b.dataset.value;activate($('#mode'),state.mode);renderAll()});
$('#position').addEventListener('input',e=>{state.position=+e.target.value;$('#position-value').textContent=Math.round(state.position/10)+'%';updateMeta();schedule()});
$('#opacity').addEventListener('input',e=>{state.opacity=+e.target.value;$('#opacity-value').textContent=state.opacity+'%';schedule()});
$('#outline').addEventListener('change',e=>{state.outline=e.target.checked;renderAll()});
function setWindow(){state.center=+$('#center').value;state.window=Math.max(1,+$('#window').value);$$('#presets button').forEach(b=>b.classList.toggle('active',+b.dataset.c===state.center&&+b.dataset.w===state.window));schedule()}
$('#center').addEventListener('change',setWindow);$('#window').addEventListener('change',setWindow);
$('#presets').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('#center').value=b.dataset.c;$('#window').value=b.dataset.w;setWindow()});
$('#columns').addEventListener('input',e=>{grid.style.setProperty('--columns',e.target.value);$('#columns-value').textContent=e.target.value});
function filterCards(){const term=$('#search').value.trim().replace(/^0+/,'');let shown=0;for(const card of $$('.card')){card.hidden=term&&!card.dataset.id.includes(term);if(!card.hidden)shown++}$('#count').textContent=`${shown} of ${state.subjects.length} subjects`;}
$('#search').addEventListener('input',()=>{filterCards();renderAll()});
function step(amount){state.position=Math.max(0,Math.min(1000,state.position+amount));$('#position').value=state.position;$('#position-value').textContent=Math.round(state.position/10)+'%';updateMeta();schedule()}
$('#previous').onclick=()=>step(-5);$('#next').onclick=()=>step(5);
document.addEventListener('keydown',e=>{if(e.target.matches('input'))return;if(e.key==='ArrowLeft')step(-5);if(e.key==='ArrowRight')step(5)});
let timer;$('#play').onclick=()=>{state.playing=!state.playing;$('#play').textContent=state.playing?'❚❚ Pause':'▶ Play';clearInterval(timer);if(state.playing)timer=setInterval(()=>{if(state.position>=1000)state.position=0;else state.position+=10;$('#position').value=state.position;$('#position-value').textContent=Math.round(state.position/10)+'%';updateMeta();schedule()},180)};

function setAlignmentControls(){const peak=state.alignment==='peak';['#position','#previous','#play','#next'].forEach(id=>$(id).disabled=peak);if(peak){state.playing=false;clearInterval(timer);$('#play').textContent='▶ Play';$('#position-value').textContent='Per subject';$('#navigation-hint').textContent='Each subject is showing its densest labeled slice.'}else{$('#position-value').textContent=Math.round(state.position/10)+'%';$('#navigation-hint').textContent='All subjects move together. Use ← and → keys for fine control.'}}

fetch('/api/subjects').then(r=>{if(!r.ok)throw Error('Server error');return r.json()}).then(data=>{state.subjects=data.subjects;if(state.subjects.some(s=>!s.peak)){state.alignment='volume';activate($('#alignment'),'volume')}$('#dataset').textContent=data.dataset;setAlignmentControls();makeCards();renderAll()}).catch(error=>{$('#status').textContent=error.message;console.error(error)});

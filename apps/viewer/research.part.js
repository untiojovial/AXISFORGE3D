// Local, self-contained reader for released experiment observations.
(() => {
  const get = name => document.getElementById('research-' + name);
  const controls = get('controls'), chooser = get('run'), step = get('step');
  let report;
  const point = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  const styles = () => getComputedStyle(document.documentElement);
  const color = name => styles().getPropertyValue('--' + name).trim();
  const values = n => Number.isFinite(n) ? Number(n.toPrecision(4)).toString() : '\u2014';

  function validate(data) {
    if (data?.schema !== 'axisforge.research.v1' || !Array.isArray(data.runs) || !data.runs.length)
      throw Error('Formato de resultados desconocido');
    for (const run of data.runs) {
      const obj = data.objects?.[run.object_id];
      if (!obj || !Array.isArray(obj.points) || !obj.points.length || !obj.points.every(point) ||
          !Array.isArray(run.rows) || !run.rows.length || typeof obj.frame !== 'string') throw Error('Geometría o trayectoria incompleta');
      for (const row of run.rows) {
        const pose = obj.candidates?.[row.candidate_id];
        if (!point(pose?.position) || !Array.isArray(pose.contacts) || !pose.contacts.every(point) ||
            !['ok','physical_failure','infrastructure_failure'].includes(row.status) ||
            !(row.utility === null || Number.isFinite(row.utility)) || !Number.isFinite(row.cost) || row.cost < 0)
          throw Error('Evaluación o coordenadas inválidas');
      }
    }
  }
  function drawGeometry(obj, pose) {
    const canvas = get('geometry'), g = canvas.getContext('2d');
    const [a,b] = get('plane').value.split('').map(v => 'XYZ'.indexOf(v));
    const all = [...obj.points, ...pose.contacts, pose.position];
    const lo = [Infinity,Infinity], hi = [-Infinity,-Infinity];
    all.forEach(p => [a,b].forEach((axis,i) => {lo[i]=Math.min(lo[i],p[axis]);hi[i]=Math.max(hi[i],p[axis]);}));
    const scale = Math.min((canvas.width-40)/Math.max(1e-8,hi[0]-lo[0]), (canvas.height-40)/Math.max(1e-8,hi[1]-lo[1]));
    const xy = p => [canvas.width/2+(p[a]-(hi[0]+lo[0])/2)*scale, canvas.height/2-(p[b]-(hi[1]+lo[1])/2)*scale];
    g.clearRect(0,0,canvas.width,canvas.height);
    g.fillStyle=color('muted'); obj.points.forEach(p=>{const [x,y]=xy(p);g.fillRect(x,y,2,2);});
    g.lineWidth=3; g.strokeStyle=color('accent');pose.contacts.forEach(p=>{g.beginPath();g.arc(...xy(p),5,0,2*Math.PI);g.stroke();});
    const [x,y]=xy(pose.position);g.strokeStyle=color('alert');g.beginPath();g.moveTo(x-6,y);g.lineTo(x+6,y);g.moveTo(x,y-6);g.lineTo(x,y+6);g.stroke();
  }
  function drawTrace(rows, n) {
    const canvas=get('trace'), g=canvas.getContext('2d'), w=canvas.width,h=canvas.height;
    const range = rows.slice(0,n).map(r=>r.utility).filter(Number.isFinite);
    const min=Math.min(0,...range),max=Math.max(1,...range);
    const x=i=>32+(w-48)*i/Math.max(1,rows.length-1), y=v=>h-26-(h-45)*(v-min)/(max-min);
    g.clearRect(0,0,w,h);g.strokeStyle=color('rule');g.beginPath();g.moveTo(32,10);g.lineTo(32,h-26);g.lineTo(w-8,h-26);g.stroke();
    g.fillStyle=color('muted');g.fillText(values(min),4,h-24);g.fillText(values(max),4,18);g.fillText('1',30,h-8);g.fillText(rows.length,w-25,h-8);
    let best=-Infinity, started=false;g.strokeStyle=color('accent');g.beginPath();
    rows.slice(0,n).forEach((r,i)=>{if(!Number.isFinite(r.utility))return;g.fillStyle=color('ink');g.fillRect(x(i)-2,y(r.utility)-2,4,4);
      if(started)g.lineTo(x(i),y(best));best=Math.max(best,r.utility);
      if(started)g.lineTo(x(i),y(best));else g.moveTo(x(i),y(best));started=true;});g.stroke();
  }
  function render() {
    const run=report.runs[Number(chooser.value)], n=Number(step.value), row=run.rows[n-1], obj=report.objects[run.object_id];
    const prefix=run.rows.slice(0,n), best=prefix.reduce((b,r)=>r.utility===null?b:Math.max(b,r.utility),-Infinity);
    drawGeometry(obj,obj.candidates[row.candidate_id]);drawTrace(run.rows,n);
    get('values').textContent=[obj.frame,run.fidelity,'Intento '+n+'/'+run.rows.length,'ID '+row.candidate_id,
      row.status,'Utilidad '+values(row.utility),'Mejor '+values(best),'Media previa '+values(row.prediction?.mean),
      'Varianza previa '+values(row.prediction?.variance),'Adquisición '+values(row.acquisition),
      'Coste acumulado '+values(prefix.reduce((s,r)=>s+r.cost,0))+' s'].join(' · ');
  }
  function setRun(){step.max=report.runs[Number(chooser.value)].rows.length;step.value='1';render();}
  get('file').addEventListener('change',async e=>{
    const file=e.target.files[0];if(!file)return;
    try{if(file.size>64*1024*1024)throw Error('El archivo supera 64 MiB');
      const next=JSON.parse(await file.text());validate(next);report=next;chooser.replaceChildren();
      report.runs.forEach((r,i)=>chooser.add(new Option([r.object_id,r.method,'semilla '+r.seed].join(' · '),String(i))));
      controls.hidden=false;get('status').textContent=report.runs.length+' trayectorias';setRun();
    }catch(error){report=null;controls.hidden=true;get('status').textContent=error.message;}
  });
  chooser.addEventListener('change',setRun);step.addEventListener('input',render);get('plane').addEventListener('change',render);
  get('csv').addEventListener('click',()=>{
    const run=report.runs[Number(chooser.value)], keys=['object_id','method','seed','fidelity','step','candidate_id','status','utility','cost_s','prior_mean','prior_variance','acquisition'];
    const cell=v=>{let s=v==null?'':String(v);if(typeof v==='string' && /^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
    const lines=[keys,...run.rows.map((r,i)=>[run.object_id,run.method,run.seed,run.fidelity,i+1,r.candidate_id,r.status,r.utility,r.cost,r.prediction?.mean,r.prediction?.variance,r.acquisition])];
    const url=URL.createObjectURL(new Blob([lines.map(l=>l.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='trayectoria-axisforge.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
})();

// Probe the existing zero-import ABI; this is not a BO or grasp evaluator.
const fs = require('fs');
(async () => {
  const bytes = fs.readFileSync(process.argv[2]);
  const {instance} = await WebAssembly.instantiate(bytes, {});
  const e = instance.exports;
  function call(request) {
    const buf = Buffer.from(JSON.stringify(request), 'utf8');
    const ptr = e.af_alloc(buf.length);
    new Uint8Array(e.memory.buffer, ptr, buf.length).set(buf);
    const packed = e.af_call(ptr, buf.length);
    const addr = Number(packed >> 32n), len = Number(packed & 0xffffffffn);
    const data = Buffer.from(new Uint8Array(e.memory.buffer, addr, len)).toString('utf8');
    e.af_free(addr, len);
    return JSON.parse(data);
  }
  for (const op of ['describe', 'list_extensions']) {
    const r = call({op});
    if (!r.ok || r.kind !== (op === 'describe' ? 'description' : 'extensions')) throw new Error(`unexpected ${op} reply`);
    console.log(`${op}: ${r.kind}, ok=${r.ok}`);
  }
  const error = call({op:'not_a_request'});
  if (error.ok !== false || typeof error.error !== 'string') throw new Error('malformed request accepted');
  console.log('malformed request: error returned');
})().catch(e => { console.error(e); process.exitCode = 1; });

// One request per fresh WASM instance. stdin/stdout is a single JSON value.
const fs = require('fs');
const crypto = require('crypto');
(async () => {
  const request = fs.readFileSync(0, 'utf8');
  const wasm = fs.readFileSync(process.argv[2]);
  const sha256 = crypto.createHash('sha256').update(wasm).digest('hex');
  const {instance} = await WebAssembly.instantiate(wasm, {});
  const e = instance.exports;
  const buf = Buffer.from(request, 'utf8');
  const ptr = e.af_alloc(buf.length);
  new Uint8Array(e.memory.buffer, ptr, buf.length).set(buf);
  const packed = e.af_call(ptr, buf.length);
  const addr = Number(packed >> 32n), len = Number(packed & 0xffffffffn);
  const result = Buffer.from(new Uint8Array(e.memory.buffer, addr, len)).toString('utf8');
  e.af_free(addr, len);
  process.stdout.write(JSON.stringify({sha256, reply: JSON.parse(result)}));
})().catch(e => { console.error(e.message); process.exitCode = 1; });

# v6 blob JSON layout correction v2 — unexecuted

Preserves exact executed v1 helper88cff in frozen-v1-executed.py.txt. ROOT's sole Git-blob POST succeeded; subsequent anonymous GET decoding failed blob-json-size-bound before any release creation. This is a JSON layout capacity error, not a native/capsule/hash mismatch. No blob POST is repeated and no original clock is reset: first1791230519.529188/end1791235919.5291885400s, operator end1791273600.

The ONLY code changes are two `limit` assignments, in decode_blob_response and fetch_verified_blob:

- Before: `max_base64 + max_base64 // 40 + 16 * 1024`
- After: `max_base64 * 11 // 10 + 16 * 1024`

GitHub base64 wrapping every60 columns introduces LF; serialized JSON escapes each LF as two bytes. CRLF every40 columns reaches10% serialized overhead. The bounded correction allows at most10% plus16KiB metadata, not arbitrary JSON expansion. For this exact1,845,748B capsule, base64 length2,461,000, max JSON2,723,484B. Streamed read remains max+1 followed by the size rejection. Decoded clean base64 length, allowed alphabet/CRLF only, repository URL, size, raw SHA256 and Git blob-header SHA1 remain unchanged. Capsule66ef..., blob30720..., all22 original SHA bindings, source/codec/helper semantics and model registrations are unchanged.

13 synthetic tests include real1,845,748B size, GitHub60col escaped LF,60col CRLF,40col LF and40col CRLF, excessive overhead rejection, wrong repository/SHA/size/unsafe chars/extra decoded content/wrong bytes, bounded exact endpoint fetching, original helper SHA preservation, and AST equivalence after restoring only those two expressions. No network/Git/POST/workflow/model inference was executed. The tests use generated non-native opaque fixtures in RAM, not raw private payload in Git.

ROOT copy recipe: replace the proposed/published logical `blob_delivery_v6.py` with the bytes of `blob_delivery_v6_v2.py`; all imports remain unchanged. Update the manifest source_bindings entry for blob_delivery_v6.py to the new SHA and re-freeze manifest/control/approval exact SHA links under the SAME already-started scope/deadline. Preserve old helper/failure and old manifest evidence. Receiver/workflow/codec/capsule bytes do not change. Before dispatch, ROOT re-reads the already-created fixed blob and verifies it with this decoder; blob availability/verification remains pending until that actual read succeeds. This patch's local tests are not relabeled as an actual GitHub readback proof.

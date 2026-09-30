# Rerun Web Viewer

These two JavaScript files come from the MIT-licensed
[`@rerun-io/web-viewer` 0.38.1 package](https://www.npmjs.com/package/@rerun-io/web-viewer/v/0.38.1).
The npm archive was verified against its published SHA-512 integrity:

```text
sha512-+COoGNxvpf9gawMdM/G2nCnaPwXTYelp41CYwtq8BRQmoJgNkcX1PjFxwq3ka0qjB0MxnbVf2H2LiYnbfSwDbQ==
```

Changes are the browser-resolvable `.js` extension on the dynamic
`./re_viewer.js` import in `index.js`, removal of trailing whitespace, and using
the Space's `media_cache.js` fetch helper for the version-pinned WASM file.
The helper preserves streaming responses and bounds persistent browser storage.
`index.js` also passes a per-viewer frame scheduler to the binding factory.
In `re_viewer.js`, animation requests/cancellations use that scheduler; decoded
video frames and resize notifications wake an idle scene. The scheduler is required; the adapter has no native scheduling fallback.
These hooks leave browser globals and video decoding unchanged. Scheduling policy lives in
`../frame_scheduler.js`, not in the generated bindings.
The Space pins Rerun, so `re_viewer.js` answers the viewer's GitHub release check
locally: no request leaves the browser and no update notice appears.
The accompanying WASM is taken from the
already installed, version-matched `gradio-rerun` wheel and checked against its
SHA-256 before serving. It is not duplicated in this repository.

Keep these files, the WASM digest in `web_assets.py`, `gradio-rerun`, and
`rerun-sdk` on the same version. Readers do not need npm or a frontend build.

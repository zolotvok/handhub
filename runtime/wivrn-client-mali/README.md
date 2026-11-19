# WiVRn client patches

Patches against the WiVRn Android client, tag `v26.9`, for running it on an ordinary
Android phone (through Monado's Cardboard driver) instead of a standalone headset. They
are derived from WiVRn and carry its licence, GPL-3.0-or-later.

| Patch | What it does | Upstreamable |
|---|---|---|
| `0001` | Requests the Vulkan `shaderClipDistance` feature only where the GPU has it. ARM Mali does not, and device creation failed with `ErrorFeatureNotPresent`. | Yes |
| `0002` | Skips the first-run questions. A phone viewer has no pointer to answer them with. | No, local |
| `0003` | Keeps the app in the recent-apps list, so leaving it does not end it. | No, local |
| `0004` | Keeps the screen on while the client runs. Nothing touches the screen in a viewer, so the phone would lock. | No, local |

## Applying and building

```bash
git clone --branch v26.9 --depth 1 https://github.com/WiVRn/WiVRn.git
cd WiVRn
git apply /path/to/handhub/runtime/wivrn-client-mali/*.patch
```

Then build the client as WiVRn's `docs/building.md` describes (Android SDK and NDK,
`rsvg-convert` and the KTX tools on the PATH, a signing key of your own). Use the same
version as the WiVRn server you run on the PC. The build installs as
`org.meumeu.wivrn.local`, beside any store version.

Create your own signing key; never commit a keystore or `gradle.properties`.

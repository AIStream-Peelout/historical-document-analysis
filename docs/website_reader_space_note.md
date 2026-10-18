# Note for the site session: the live playground moves to a Hugging Face ZeroGPU Space

Decision (user, 2026-09-10): the realtime "read this fragment" playground runs on a ZeroGPU
Space, not on the Studio's LM Studio. The site only needs a link.

- Space: `https://huggingface.co/spaces/isaacmg/genizah-reader` (private until the user flips it
  public in the Space settings; the link works for the owner meanwhile).
- Model served: `isaacmg/qwen3-vl-8b-hebrew-v20a-merged` (public).
- Tasks: transcribe the page · lines with boxes (boxes stream in as lines complete) · find a
  phrase. Same prompts as training. Visitors upload their own image.
- Quotas are per visitor, enforced by Hugging Face (anonymous 2 min GPU/day, free account 5 min,
  PRO 40 min); a grounded page read is ~20–40 s. So the link needs no rate limiting on our side.
- Suggested placement: a "Try it live (experimental)" button on the document page next to the
  offline read, opening the Space in a new tab; keep the experimental copy from
  `website_grounding_playground_prompt.md`. Nothing in the backend changes; the SSE endpoint
  design from that prompt is shelved.
- Rights: visitors' uploads never touch our data; the site's own images stay on the site.

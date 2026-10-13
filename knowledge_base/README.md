# Knowledge base

Markdown playbooks retrieved by the copilot (stage 6). Each file has YAML front
matter (`id`, `title`, `tags`, `references`) and `##` sections; each section is
one retrievable chunk with the stable id `<id>#<section-slug>`. Numbered list
items are treated as candidate investigation/response steps, and sections whose
heading contains *containment*, *response*, *recovery* or *approval* are always
marked as requiring human approval.

These playbooks are **illustrative research artefacts**. They summarise common
OT incident-handling practice (NIST SP 800-82, NIST SP 800-61, IEC 62443) and
map behaviours to MITRE ATT&CK for ICS technique identifiers, but they are not a
substitute for a site's approved standard operating procedures. Validate and
adapt them with plant operations and safety engineering before any real use.

The retrieval index is built from these files only, so it never contains
telemetry and cannot leak evaluation data. File hashes are recorded with every
run (`knowledge_base.json`).

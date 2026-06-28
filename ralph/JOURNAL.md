# Ralph Journal

Append-only log of what each run did. Newest entries go at the bottom. The next
run reads the last few entries to avoid repeating work.

Format per entry:

```
## YYYY-MM-DD HH:MM — <short title>
- Found: <the bug/gap and how it was detected>
- Fix:   <what changed>
- Verify: <build/test command + result>
- Commit: <short SHA>
```

---
<!-- Ralph appends below this line. -->

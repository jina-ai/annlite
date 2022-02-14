

## Release MatchMerger Addon

```bash
$ docker buildx build --platform linux/amd64,linux/arm64 --push \
  -t numb3r3/pqlite-merger:v0.2.3 \
  -t numb3r3/pqlite-merger:latest .
```

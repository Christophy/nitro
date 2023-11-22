# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```
$ USE_SSH=true yarn deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

### Preview URL, Pre-release and Publishing Documentation

When a PR is created, the preview URL will be automatically commented on the PR. For staging or pre-release documentation, use the following domains [https://pre-release-nitro.jan.ai/](https://pre-release-nitro.jan.ai/)

To officially publish documentation, create a tag in the format `vx.y.z-docs` (e.g., `v0.1.1-docs`) on the `main` branch. The documentation will then be published to [https://nitro.jan.ai/](https://nitro.jan.ai/)

### Additional Plugins
- @docusaurus/theme-live-codeblock
- [Redocusaurus](https://redocusaurus.vercel.app/): manually upload swagger files at `/openapi/OpenAPISpec.json`
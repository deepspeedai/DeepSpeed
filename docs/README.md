# DeepSpeed Documentation

This directory includes the source code for the website and documentation of DeepSpeed. The `code-docs/` directory is used to build [deepspeed.readthedocs.io](https://deepspeed.readthedocs.io/en/latest/).

[deepspeed.ai](https://www.deepspeed.ai/) is the recommended way to read all DeepSpeed documentation. Directly viewing the Markdown files in this directory will not include images and other features.

## Building the documentation locally
You can serve the DeepSpeed website locally. This is especially useful for development.

### Prerequisites
The DeepSpeed website relies on [Jekyll](https://jekyllrb.com/). There are several [guides for installation](https://jekyllrb.com/docs/installation/). The instructions below assume you are in an Ubuntu environment and have been tested on WSL.

First ensure that you have the necessary packages (e.g., `make` and `zlib`).
```
sudo apt-get install build-essential zlib1g-dev ruby-full
```

Add these lines to your `.bashrc` or equivalent to ensure you have permissions to install Ruby packages without `sudo`.
```
export GEM_HOME="$HOME/gems"
export PATH="$HOME/gems/bin:$PATH"
```
Don't forget to `source ~/.bashrc` afterwards 😊.


Now we can install Jekyll and [Bundler](https://bundler.io/):
```
gem install jekyll bundler
```

### Start a local webserver
We now need to install the required Ruby packages for the website.

**NOTE**: you should change to this folder (i.e., `docs`) before running the installation command to avoid this [error](https://stackoverflow.com/questions/10012181/bundle-install-returns-could-not-locate-gemfile/35157872):

> Could not locate Gemfile

**NOTE**: this step frequently hangs when connected to a VPN (including MSVPN). Simply disconnect for the package installation.


```
bundle install
```

You can now start a local webserver via:
```
bundle exec jekyll serve
```
The website should now be accessible at [http://localhost:4000](http://localhost:4000)


## Update the Readthedocs.io API documentation
Use the following steps to update the public API documentation.

1. Make your documentation changes and push them to the rtd-staging branch. This will rebuild the docs in the staging branch.
**NOTE**: It is acceptable to force push to this branch to overwrite previous changes.
2. View the result of the result of the build [here](https://readthedocs.org/projects/deepspeed/builds/)
3. Once the build is complete view the newly modified API documentation [here](https://deepspeed.readthedocs.io/en/rtd-staging/)
4. Once you are satisfied with the changes create a new branch off of rtd-staging to push into master.

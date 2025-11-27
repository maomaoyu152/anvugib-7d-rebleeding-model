# install_packages.R
# Helper script to install R packages required for the Shiny application
# of the ANVUGIB 7-day rebleeding risk prediction model.

required_packages <- c(
  "shiny",
  "dplyr",
  "ggplot2",
  "DT",
  "readr"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

for (pkg in required_packages) {
  install_if_missing(pkg)
}

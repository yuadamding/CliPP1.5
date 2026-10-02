# Experimental CPU recipe: image execution has not been qualified for this revision.
FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
RUN CLIPP_USE_CUDA=0 python -m pip wheel --wheel-dir /wheels .

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends r-base-core r-cran-data.table && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels clipp15 && rm -rf /wheels
RUN mkdir /work && chmod 1777 /work
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["clipp"]
CMD ["--help"]

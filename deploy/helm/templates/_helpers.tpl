{{- define "deepinferforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepinferforge.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "deepinferforge.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

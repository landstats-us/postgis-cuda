/*
 * gpu_spatial: thin client for gpujoind (GPU spatial join daemon).
 * Layers are loaded into GPU memory by the daemon and stay resident across sessions; the functions here send
 * requests and stream results back as set-returning functions. No CUDA or PostGIS linkage in the backend.
 */
#include "postgres.h"

#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "fmgr.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/tuplestore.h"

#include "protocol.h"

PG_MODULE_MAGIC;

static char *daemon_addr = NULL;
static char *daemon_token = NULL;
static char *last_stats = NULL; /* JSON of the last join/cluster, in TopMemoryContext */

void _PG_init(void);

void
_PG_init(void)
{
	DefineCustomStringVariable("gpu_spatial.daemon",
							   "Address of gpujoind: unix:/path/to/socket or tcp:host:port.",
							   NULL, &daemon_addr, "unix:/var/run/gpu_spatial/gpujoind.sock",
							   PGC_SUSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("gpu_spatial.token",
							   "Shared secret sent to gpujoind (must match its --token-file).",
							   NULL, &daemon_token, "",
							   PGC_SUSET, GUC_SUPERUSER_ONLY | GUC_NO_SHOW_ALL, NULL, NULL, NULL);
	MarkGUCPrefixReserved("gpu_spatial");
}

/* ------------------------------------------------------------------------------------------- connection */

typedef struct
{
	int			fd;
	char		buf[1 << 20];
	size_t		pos,
				len;
} GspConn;

static int
gsp_connect(void)
{
	const char *a = daemon_addr ? daemon_addr : "";
	int			fd = -1;

	if (strncmp(a, "unix:", 5) == 0)
	{
		struct sockaddr_un sa;

		memset(&sa, 0, sizeof(sa));
		sa.sun_family = AF_UNIX;
		strlcpy(sa.sun_path, a + 5, sizeof(sa.sun_path));
		fd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (fd < 0 || connect(fd, (struct sockaddr *) &sa, sizeof(sa)) != 0)
		{
			int			e = errno;

			if (fd >= 0)
				close(fd);
			ereport(ERROR, (errcode(ERRCODE_CONNECTION_FAILURE),
							errmsg("could not connect to gpujoind at \"%s\": %s", a, strerror(e)),
							errhint("Start gpujoind or set gpu_spatial.daemon.")));
		}
	}
	else if (strncmp(a, "tcp:", 4) == 0)
	{
		char		host[256];
		const char *colon = strrchr(a + 4, ':');
		struct addrinfo hints, *res = NULL;
		int			rc;

		if (!colon || colon - (a + 4) >= (int) sizeof(host))
			ereport(ERROR, (errmsg("bad gpu_spatial.daemon \"%s\"", a)));
		memcpy(host, a + 4, colon - (a + 4));
		host[colon - (a + 4)] = '\0';
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		rc = getaddrinfo(host, colon + 1, &hints, &res);
		if (rc != 0)
			ereport(ERROR, (errmsg("could not resolve \"%s\": %s", host, gai_strerror(rc))));
		fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
		if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) != 0)
		{
			int			e = errno;

			freeaddrinfo(res);
			if (fd >= 0)
				close(fd);
			ereport(ERROR, (errcode(ERRCODE_CONNECTION_FAILURE),
							errmsg("could not connect to gpujoind at \"%s\": %s", a, strerror(e))));
		}
		freeaddrinfo(res);
	}
	else
		ereport(ERROR, (errmsg("gpu_spatial.daemon must start with unix: or tcp:")));
	return fd;
}

static void
gsp_write_all(int fd, const char *p, size_t n)
{
	while (n > 0)
	{
		ssize_t		k = send(fd, p, n, MSG_NOSIGNAL);

		if (k < 0)
		{
			if (errno == EINTR)
				continue;
			ereport(ERROR, (errmsg("gpujoind: send failed: %m")));
		}
		p += k;
		n -= k;
	}
}

static void
gsp_fill(GspConn *c)
{
	if (c->pos > 0 && c->pos < c->len)
		memmove(c->buf, c->buf + c->pos, c->len - c->pos);
	c->len -= c->pos;
	c->pos = 0;
	for (;;)
	{
		struct pollfd pfd = {c->fd, POLLIN, 0};
		int			r = poll(&pfd, 1, 100);
		ssize_t		k;

		CHECK_FOR_INTERRUPTS();
		if (r < 0)
		{
			if (errno == EINTR)
				continue;
			ereport(ERROR, (errmsg("gpujoind: poll failed: %m")));
		}
		if (r == 0)
			continue;
		k = read(c->fd, c->buf + c->len, sizeof(c->buf) - c->len);
		if (k == 0)
			ereport(ERROR, (errmsg("gpujoind closed the connection unexpectedly")));
		if (k < 0)
		{
			if (errno == EINTR || errno == EAGAIN)
				continue;
			ereport(ERROR, (errmsg("gpujoind: read failed: %m")));
		}
		c->len += k;
		return;
	}
}

static const char *
gsp_take(GspConn *c, size_t n)
{
	const char *p;

	Assert(n <= sizeof(c->buf));
	while (c->len - c->pos < n)
		gsp_fill(c);
	p = c->buf + c->pos;
	c->pos += n;
	return p;
}

static uint32
gsp_u32(GspConn *c)
{
	uint32		v;

	memcpy(&v, gsp_take(c, 4), 4);
	return v;
}

/* palloc'd, NUL-terminated */
static char *
gsp_str(GspConn *c)
{
	uint32		n = gsp_u32(c);
	char	   *s = palloc(n + 1);
	uint32		got = 0;

	while (got < n)
	{
		uint32		k = Min(n - got, (uint32) (sizeof(c->buf) / 2));

		memcpy(s + got, gsp_take(c, k), k);
		got += k;
	}
	s[n] = '\0';
	return s;
}

static void
gsp_send_request(GspConn *c, uint32 op, int nargs, const char **args)
{
	StringInfoData b;
	uint32		h[3] = {GSP_MAGIC, op, (uint32) nargs + 1};
	const char *tok = daemon_token ? daemon_token : "";
	uint32		tl = (uint32) strlen(tok);

	initStringInfo(&b);
	appendBinaryStringInfo(&b, (const char *) h, sizeof(h));
	appendBinaryStringInfo(&b, (const char *) &tl, 4);
	appendBinaryStringInfo(&b, tok, tl);
	for (int i = 0; i < nargs; i++)
	{
		uint32		l = (uint32) strlen(args[i]);

		appendBinaryStringInfo(&b, (const char *) &l, 4);
		appendBinaryStringInfo(&b, args[i], l);
	}
	gsp_write_all(c->fd, b.data, b.len);
	pfree(b.data);
}

static void
gsp_check_status(GspConn *c)
{
	if (gsp_u32(c) != GSP_OK)
		ereport(ERROR, (errmsg("gpujoind: %s", gsp_str(c))));
}

static GspConn *
gsp_open(void)
{
	GspConn    *c = palloc(sizeof(GspConn));

	c->pos = c->len = 0;
	c->fd = gsp_connect();
	return c;
}

/* request returning one JSON document */
static Datum
gsp_json_call(uint32 op, int nargs, const char **args)
{
	GspConn    *c = gsp_open();
	char	   *json = NULL;

	PG_TRY();
	{
		gsp_send_request(c, op, nargs, args);
		gsp_check_status(c);
		json = gsp_str(c);
	}
	PG_FINALLY();
	{
		close(c->fd);
	}
	PG_END_TRY();
	return DirectFunctionCall1(jsonb_in, CStringGetDatum(json));
}

static const char *
text_arg(FunctionCallInfo fcinfo, int i, const char *dflt)
{
	return PG_ARGISNULL(i) ? dflt : text_to_cstring(PG_GETARG_TEXT_PP(i));
}

static void
set_last_stats(const char *json)
{
	if (last_stats)
		pfree(last_stats);
	last_stats = MemoryContextStrdup(TopMemoryContext, json);
}

/* ------------------------------------------------------------------------------------------- functions */

PG_FUNCTION_INFO_V1(gpu_load);
Datum
gpu_load(PG_FUNCTION_ARGS)
{
	Oid			relid;
	char	   *relname;
	char		sub[16];
	const char *args[7];

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR, (errmsg("layer and rel are required")));
	relid = PG_GETARG_OID(1);
	relname = get_rel_name(relid);
	if (!relname)
		ereport(ERROR, (errmsg("relation %u does not exist", relid)));
	snprintf(sub, sizeof(sub), "%d", PG_ARGISNULL(5) ? 0 : PG_GETARG_INT32(5));
	args[0] = text_arg(fcinfo, 0, "");
	args[1] = quote_qualified_identifier(get_namespace_name(get_rel_namespace(relid)), relname);
	args[2] = text_arg(fcinfo, 2, "id");
	args[3] = text_arg(fcinfo, 3, "geom");
	args[4] = text_arg(fcinfo, 4, "");
	args[5] = sub;
	args[6] = (PG_ARGISNULL(6) || PG_GETARG_BOOL(6)) ? "1" : "0";
	PG_RETURN_DATUM(gsp_json_call(GSP_OP_LOAD, 7, args));
}

PG_FUNCTION_INFO_V1(gpu_load_query);
Datum
gpu_load_query(PG_FUNCTION_ARGS)
{
	char		sub[16];
	const char *args[4];

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR, (errmsg("layer and query are required")));
	snprintf(sub, sizeof(sub), "%d", PG_ARGISNULL(2) ? 0 : PG_GETARG_INT32(2));
	args[0] = text_arg(fcinfo, 0, "");
	args[1] = text_arg(fcinfo, 1, "");
	args[2] = sub;
	args[3] = (PG_ARGISNULL(3) || PG_GETARG_BOOL(3)) ? "1" : "0";
	PG_RETURN_DATUM(gsp_json_call(GSP_OP_LOAD_QUERY, 4, args));
}

PG_FUNCTION_INFO_V1(gpu_unload);
Datum
gpu_unload(PG_FUNCTION_ARGS)
{
	const char *args[1] = {text_to_cstring(PG_GETARG_TEXT_PP(0))};
	Jsonb	   *j = DatumGetJsonbP(gsp_json_call(GSP_OP_DROP, 1, args));
	JsonbValue	v;

	PG_RETURN_BOOL(JsonbExtractScalar(&j->root, &v) && v.type == jbvBool && v.val.boolean);
}

PG_FUNCTION_INFO_V1(gpu_status);
Datum
gpu_status(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(gsp_json_call(GSP_OP_STATUS, 0, NULL));
}

PG_FUNCTION_INFO_V1(gpu_layers_json);
Datum
gpu_layers_json(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(gsp_json_call(GSP_OP_LIST, 0, NULL));
}

PG_FUNCTION_INFO_V1(gpu_last_stats);
Datum
gpu_last_stats(PG_FUNCTION_ARGS)
{
	if (!last_stats)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(last_stats)));
}

/* Read result frames until the end marker; emit one tuple per row via the callback. */
typedef void (*row_fn) (GspConn *c, ReturnSetInfo *rsinfo);

static void
stream_rows(GspConn *c, ReturnSetInfo *rsinfo, row_fn fn)
{
	for (;;)
	{
		uint32		n = gsp_u32(c);

		if (n == 0)
			break;
		if (n == GSP_FRAME_ERR)
			ereport(ERROR, (errmsg("gpujoind: %s", gsp_str(c))));
		for (uint32 i = 0; i < n; i++)
			fn(c, rsinfo);
		CHECK_FOR_INTERRUPTS();
	}
	set_last_stats(gsp_str(c));
}

/* Text datums are built in static buffers: tuplestore_putvalues copies them into the tuple. */
typedef union
{
	struct varlena v;
	char		raw[VARHDRSZ + 65536];
} IdBuf;
static IdBuf idbuf_a,
			idbuf_b;

static Datum
read_id(GspConn *c, IdBuf *b)
{
	uint16		l;

	memcpy(&l, gsp_take(c, 2), 2);
	memcpy(VARDATA(&b->v), gsp_take(c, l), l);
	SET_VARSIZE(&b->v, l + VARHDRSZ);
	return PointerGetDatum(&b->v);
}

static void
join_row(GspConn *c, ReturnSetInfo *rsinfo)
{
	Datum		v[3];
	bool		nulls[3] = {false, false, false};
	double		area;

	v[0] = read_id(c, &idbuf_a);
	v[1] = read_id(c, &idbuf_b);
	memcpy(&area, gsp_take(c, 8), 8);
	if (isnan(area))
		nulls[2] = true;
	else
		v[2] = Float8GetDatum(area);
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
}

static void
cluster_row(GspConn *c, ReturnSetInfo *rsinfo)
{
	Datum		v[2];
	bool		nulls[2] = {false, false};
	uint32		k;

	v[0] = read_id(c, &idbuf_a);
	memcpy(&k, gsp_take(c, 4), 4);
	v[1] = Int32GetDatum((int32) k);
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
}

static void
relate_row(GspConn *c, ReturnSetInfo *rsinfo)
{
	Datum		v[3];
	bool		nulls[3] = {false, false, false};

	v[0] = read_id(c, &idbuf_a);
	v[1] = read_id(c, &idbuf_b);
	v[2] = PointerGetDatum(cstring_to_text_with_len(gsp_take(c, 9), 9));
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
	pfree(DatumGetPointer(v[2]));
}

static void
srf_call(FunctionCallInfo fcinfo, uint32 op, int nargs, const char **args, row_fn fn)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	GspConn    *c;

	InitMaterializedSRF(fcinfo, 0);
	c = gsp_open();
	PG_TRY();
	{
		gsp_send_request(c, op, nargs, args);
		gsp_check_status(c);
		stream_rows(c, rsinfo, fn);
	}
	PG_FINALLY();
	{
		close(c->fd);	/* on cancel the daemon sees the closed socket and stops */
		pfree(c);
	}
	PG_END_TRY();
}

PG_FUNCTION_INFO_V1(gpu_join);
Datum
gpu_join(PG_FUNCTION_ARGS)
{
	const char *args[4];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = PG_GETARG_BOOL(2) ? "1" : "0";
	args[3] = text_to_cstring(PG_GETARG_TEXT_PP(3));
	srf_call(fcinfo, GSP_OP_JOIN, 4, args, join_row);
	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(gpu_cluster_intersecting);
Datum
gpu_cluster_intersecting(PG_FUNCTION_ARGS)
{
	const char *args[1];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	srf_call(fcinfo, GSP_OP_CLUSTER, 1, args, cluster_row);
	return (Datum) 0;
}

/* Run a join without returning rows: row count (after expansion) and summed area as JSON. */
PG_FUNCTION_INFO_V1(gpu_join_stats);
Datum
gpu_join_stats(PG_FUNCTION_ARGS)
{
	const char *args[5];
	GspConn    *c;
	char	   *json = NULL;

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = PG_GETARG_BOOL(2) ? "1" : "0";
	args[3] = text_to_cstring(PG_GETARG_TEXT_PP(3));
	args[4] = "stats";
	c = gsp_open();
	PG_TRY();
	{
		gsp_send_request(c, GSP_OP_JOIN, 5, args);
		gsp_check_status(c);
		for (;;)
		{
			uint32		n = gsp_u32(c);

			if (n == 0)
				break;
			if (n == GSP_FRAME_ERR)
				ereport(ERROR, (errmsg("gpujoind: %s", gsp_str(c))));
			ereport(ERROR, (errmsg("gpujoind: unexpected rows in stats mode")));
		}
		json = gsp_str(c);
		set_last_stats(json);
	}
	PG_FINALLY();
	{
		close(c->fd);
		pfree(c);
	}
	PG_END_TRY();
	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json)));
}

PG_FUNCTION_INFO_V1(gpu_relate);
Datum
gpu_relate(PG_FUNCTION_ARGS)
{
	const char *args[4];

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR, (errmsg("layers are required")));
	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = text_arg(fcinfo, 2, "");
	args[3] = text_arg(fcinfo, 3, "all");
	srf_call(fcinfo, GSP_OP_RELATE, 4, args, relate_row);
	return (Datum) 0;
}

/* (a_id, b_id) rows of a DWITHIN stream (the area field is always NaN) */
static void
pair_row(GspConn *c, ReturnSetInfo *rsinfo)
{
	Datum		v[2];
	bool		nulls[2] = {false, false};

	v[0] = read_id(c, &idbuf_a);
	v[1] = read_id(c, &idbuf_b);
	(void) gsp_take(c, 8);
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
}

static const char *
f8_arg(FunctionCallInfo fcinfo, int i)
{
	double		v = PG_GETARG_FLOAT8(i);

	if (!isfinite(v))
		ereport(ERROR, (errmsg("distances must be finite")));
	return psprintf("%.17g", v);
}

/* Run a DWITHIN request in stats mode: no rows, the JSON stats */
static Datum
dwithin_stats(FunctionCallInfo fcinfo, const char **args)
{
	GspConn    *c;
	char	   *json = NULL;

	args[6] = "stats";
	c = gsp_open();
	PG_TRY();
	{
		gsp_send_request(c, GSP_OP_DWITHIN, 7, args);
		gsp_check_status(c);
		for (;;)
		{
			uint32		n = gsp_u32(c);

			if (n == 0)
				break;
			if (n == GSP_FRAME_ERR)
				ereport(ERROR, (errmsg("gpujoind: %s", gsp_str(c))));
			ereport(ERROR, (errmsg("gpujoind: unexpected rows in stats mode")));
		}
		json = gsp_str(c);
		set_last_stats(json);
	}
	PG_FINALLY();
	{
		close(c->fd);
		pfree(c);
	}
	PG_END_TRY();
	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json)));
}

/* gpu_dwithin(a, b, distance, self_pairs): ST_DWithin(a::geography, b::geography, distance) */
PG_FUNCTION_INFO_V1(gpu_dwithin);
Datum
gpu_dwithin(PG_FUNCTION_ARGS)
{
	const char *args[6];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = "dwithin";
	args[3] = f8_arg(fcinfo, 2);
	args[4] = "0";
	args[5] = text_to_cstring(PG_GETARG_TEXT_PP(3));
	srf_call(fcinfo, GSP_OP_DWITHIN, 6, args, pair_row);
	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(gpu_dwithin_stats);
Datum
gpu_dwithin_stats(PG_FUNCTION_ARGS)
{
	const char *args[7];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = "dwithin";
	args[3] = f8_arg(fcinfo, 2);
	args[4] = "0";
	args[5] = text_to_cstring(PG_GETARG_TEXT_PP(3));
	return dwithin_stats(fcinfo, args);
}

/* gpu_intersects_buffered(a, b, buffer_a, buffer_b, self_pairs):
 * ST_Intersects(ST_Buffer(a::geography, buffer_a), ST_Buffer(b::geography, buffer_b)) */
PG_FUNCTION_INFO_V1(gpu_intersects_buffered);
Datum
gpu_intersects_buffered(PG_FUNCTION_ARGS)
{
	const char *args[6];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = "buffer";
	args[3] = f8_arg(fcinfo, 2);
	args[4] = f8_arg(fcinfo, 3);
	args[5] = text_to_cstring(PG_GETARG_TEXT_PP(4));
	srf_call(fcinfo, GSP_OP_DWITHIN, 6, args, pair_row);
	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(gpu_intersects_buffered_stats);
Datum
gpu_intersects_buffered_stats(PG_FUNCTION_ARGS)
{
	const char *args[7];

	args[0] = text_to_cstring(PG_GETARG_TEXT_PP(0));
	args[1] = text_to_cstring(PG_GETARG_TEXT_PP(1));
	args[2] = "buffer";
	args[3] = f8_arg(fcinfo, 2);
	args[4] = f8_arg(fcinfo, 3);
	args[5] = text_to_cstring(PG_GETARG_TEXT_PP(4));
	return dwithin_stats(fcinfo, args);
}

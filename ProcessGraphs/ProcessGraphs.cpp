#include "ProcessGraphs.h"
#include <unistd.h>
#include <sys/stat.h>

Graph::Graph(/* args */)
{
    n = m = maxdeg = 0;
    adj = NULL;
    data = deg = NULL;
}

Graph::~Graph()
{
    if (adj != NULL) delete[] adj;
    if (data != NULL) delete[] data;
    if (deg != NULL) delete[] deg;
}

void Graph::readGraph_directed(const char *str)
{
	printf("Reading file:%s\n", str);
	FILE *in = fopen(str, "r");
	if (in == NULL) {
		printf("No such the input file\n");
		exit(1);
	}

	char buf[128];
	while (fgets(buf,128,in) != NULL)
		if (*buf <= '9' && *buf >= '0') break;
	sscanf(buf, "%d %d", &n, &m);
	printf("n=%d, m=%d\n", n, m);

	if (data == NULL) data = new int[m]();
	if (adj == NULL) adj = new int *[n];
	if (deg == NULL) deg = new int[n]();
	
	int u, v, cnt = 0, last_u = -1;
	for (int i = 0; i < m; ++i) {
		char *r = fgets(buf, 128, in);
		sscanf(buf, "%d %d", &u, &v);
		if (u == v) continue;
		assert(u < n && u >= 0);
		assert(v < n && v >= 0);

		data[cnt++] = v;
		deg[u]++;
	}
	fclose(in);
	adj[0] = data; maxdeg = deg[0];
	for (int i = 0; i < n - 1; ++i){
		adj[i+1] = adj[i] + deg[i];
		maxdeg = max(maxdeg, deg[i+1]);
	}
	printf("maxdeg=%d\n", maxdeg);
	printf("Reading done\n");
}

void Graph::writeGraph_directed(const char *str)
{
    FILE *out = fopen(str, "w");
	if (!out)
	{
		printf("No such the outfile: %s\n", str);
		exit(1);
	}

    fprintf(out, "%d %d\n", n, m);

    for (int i = 0; i < n; ++i)
    {
        int d = deg[i];
        for (int j = 0; j < d; ++j)
            fprintf(out, "%d %d\n", i, adj[i][j]);
    }
    fclose(out);
}

void Graph::readGraph_undirected(const char *str)
{
    printf("Reading the graph\nFileName:%s\n", str);
	FILE *in = fopen(str, "r");
	if (in == NULL) {
		printf("No such the input file\n");
		exit(1);
	}
	char buf[128];

	while (fgets(buf, 128, in) != NULL)
		if (*buf <= '9' && *buf >= '0') break;

	sscanf(buf, "%d %d", &n, &m);
	printf("n=%d, m=%d\n", n, m);

	if (adj == NULL) adj = new int *[n];
	if (deg == NULL) deg = new int[n]();

	int u, v; m = 0;
	while (fgets(buf, 128, in) != NULL)
	{
		sscanf(buf, "%d %d", &u, &v);
		assert(u < n && u > -1);
		assert(v < n && v > -1);
		deg[u]++;
		deg[v]++;
		m += 2;
	}
	fclose(in);
	if (data == NULL) data = new int[m]();
	m = 0;
	for (int i = 0; i < n; ++i) {
		adj[i] = data + m;
		m += deg[i];
		maxdeg = max(maxdeg, deg[i]);
		deg[i] = 0;
	}
	
	in = fopen(str, "r");
	while (fgets(buf, 128, in) != NULL)
		if (*buf <= '9' && *buf >= '0') break;
	while (fgets(buf, 128, in) != NULL)
	{
		sscanf(buf, "%d %d", &u, &v);
		adj[u][deg[u]++] = v;
		adj[v][deg[v]++] = u;
	}
	fclose(in);
	printf("Reading done\n");

}

void Graph::writeGraph_undirected(const char *str)
{
    FILE *out = fopen(str, "w");
	if (!out)
	{
		printf("No such the outfile: %s\n", str);
		exit(1);
	}
    fprintf(out, "%d %d\n", n, m/2);
    for (int i = 0; i < n; ++i)
    {
        int d = deg[i];
        for (int j = 0; j < d; ++j)
        {
            int v = adj[i][j];
            if (i < v) fprintf(out, "%d %d\n", i, v);
        }
    }
    fclose(out);
}

void Graph::readGraph_renumbering(const char *str)
{
    FILE *in = fopen(str, "r");
	if (!in)
	{
		printf("No such the infile: %s\n", str);
		exit(1);
	}

	char A[512], *p;
	while (fgets(A, 512, in))
		if (A[0] >= '0' && A[0] <= '9') break;
	int row, column, edges;
	sscanf(A,"%d %d %d", &row, &column, &edges);

	vector< pair<int,int> > datas;
	datas.reserve(edges);
	int x, y;
    n=0, m=0;
	while (fgets(A, 512, in))
	{
		sscanf(A,"%d %d", &x, &y);
		if (x == y) continue;
		datas.push_back(pair<int,int>(x,y));
		datas.push_back(pair<int,int>(y,x));
		n = max(n, max(x, y));
	}
    fclose(in);
	printf("n=%d, m=%ld\n", n, datas.size());

	int id = 0, *ids = new int[n+1];
	memset(ids, -1, sizeof(int)* (n+1));
	m = datas.size();
	for (int i = 0; i < m; ++i) {
		x = datas[i].first;
		y = datas[i].second;
		if (ids[x] < 0) ids[x] = id++;
		x = ids[x];
		if (ids[y] < 0) ids[y] = id++;
		y = ids[y];
		datas[i].first = x;
		datas[i].second = y;
	}
	sort(datas.begin(), datas.end());

	n = id; m = 0; x = -1; y = -1;
	for (size_t i = 0; i < datas.size(); i++) {
		int a = datas[i].first, b = datas[i].second;
		if (x == a && y == b) continue;
		datas[m].first = a;
		datas[m++].second = b;
		x = a; y = b;
	}

    if (adj == NULL) adj = new int*[n];
    if (deg == NULL) deg = new int[n]();
    if (data == NULL) data = new int[m];

    for (int i = 0; i < m; ++i)
    {
        x = datas[i].first;
        data[i] = datas[i].second;
        deg[x]++;
    }
    adj[0] = data; maxdeg = deg[0];
    for (int i = 0; i < n - 1; ++i) {
        adj[i+1] = adj[i] + deg[i];
        maxdeg = max(maxdeg, deg[i+1]);
    }
}

void Graph::readGraph_clq(const char *str)
{
	printf("Reading graph\nFileName:%s\n", str);
	FILE *in = fopen(str, "r");
	if (in == NULL) {
		printf("No such the input file\n");
		exit(1);
	}
	char buf[512];

	while (fgets(buf, 512, in) != NULL)
		if (*buf == 'p') break;

	sscanf(buf+6, "%d %d", &n, &m);
	printf("n=%d, m=%d\n", n, m);

	if (adj == NULL) adj = new int *[n];
	if (deg == NULL) deg = new int[n]();

	int u, v; m = 0;
	while (fgets(buf, 128, in) != NULL)
	{
		if (buf[0] != 'e') continue;
		sscanf(buf + 2, "%d %d", &u, &v); u--; v--;
		assert(u < n && u > -1);
		assert(v < n && v > -1);
		deg[u]++;
		deg[v]++;
		m += 2;
	}
	fclose(in);
	if (data == NULL) data = new int[m]();
	m = 0;
	for (int i = 0; i < n; ++i) {
		adj[i] = data + m;
		m += deg[i];
		maxdeg = max(maxdeg, deg[i]);
		deg[i] = 0;
	}
	
	in = fopen(str, "r");
	while (fgets(buf, 512, in) != NULL)
		if (*buf == 'p') break;
	while (fgets(buf, 128, in) != NULL)
	{
		if (buf[0] != 'e') continue;
		sscanf(buf + 2, "%d %d", &u, &v); u--; v--;
		adj[u][deg[u]++] = v;
		adj[v][deg[v]++] = u;
	}
	fclose(in);
	printf("Reading done\n");
}

void Graph::writeGraph_clq(const char *str)
{
	FILE *out = fopen(str, "w");
	if (!out)
	{
		printf("No such the outfile: %s\n", str);
		exit(1);
	}
    fprintf(out, "p edge %d %d\n", n, m/2);
    for (int i = 0; i < n; ++i)
    {
        int d = deg[i];
        for (int j = 0; j < d; ++j)
        {
            int v = adj[i][j];
            if (i > v) fprintf(out, "e %d %d\n", i+1, v+1);
        }
    }
    fclose(out);
}

void Graph::writeGraph_DirWithComma(const char *str)
{
	FILE *out = fopen(str, "w");
	if (!out)
	{
		printf("No such the outfile: %s\n", str);
		exit(1);
	}
    fprintf(out, "%d\n%d\n", n, m);
    for (int i = 0; i < n; ++i)
    {
        int d = deg[i];
        for (int j = 0; j < d; ++j)
        {
            int v = adj[i][j];
            fprintf(out, "%d,%d\n", i, v);
        }
    }
    fclose(out);
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: ./processgh filepath <readformat> <writeformat>\n");
        printf("readformat: -rud -rd -rclq -rrn\n");
        printf("writeformat: -wud -wd -wclq -wcm -non\n");
        return 1;
    }
    const char *defaultReadFormat = "-rud";
    const char *defaultWriteFormat = "-wud";
	char *file = argv[1];
	char outfile[128]="";
    const char *r = argc > 2 ? argv[2] : defaultReadFormat;
    const char *w = argc > 3 ? argv[3] : defaultWriteFormat;

    Graph g;

    if (strcmp(r, "-rud") == 0)
        g.readGraph_undirected(file);
    else if (strcmp(r, "-rd") == 0)
        g.readGraph_directed(file);
    else if (strcmp(r, "-rrn") == 0)
        g.readGraph_renumbering(file);
	else if (strcmp(r, "-rclq") == 0)
		g.readGraph_clq(file);

	//set outpath and outfile name
    const char *filepath = "Output/";
    if (access(filepath, 0) == -1) {
        mkdir(filepath, 0777);
    }
    char *p = strrchr(file, '/');
    if (p == NULL) p = file;
    else p+=1;
    strcat(outfile, filepath);
    int len = strlen(p) < 4 ? 0 : (strlen(p) - 4);
    strncat(outfile, p, len);

    if (strcmp(w, "-wud") == 0) {
		strcat(outfile, "-undirect.txt");
        g.writeGraph_undirected(outfile);
	}
    else if (strcmp(w, "-wd") == 0) {
		strcat(outfile, "-direct.txt");
        g.writeGraph_directed(outfile);
	}
	else if (strcmp(w, "-wclq") == 0) {
		strcat(outfile, ".clq");
        g.writeGraph_clq(outfile);
	}
	else if (strcmp(w, "-wcm") == 0) {
		strcat(outfile, "-comma.txt");
        g.writeGraph_DirWithComma(outfile);
	}

    return 0;
}
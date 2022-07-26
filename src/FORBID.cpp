#include <iostream>

#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort, std::stable_sort
#include <float.h>

#include "FORBID.hpp"

double maxOfLeft(Coord const &p1, const Size &s1, Coord const &p2, const Size &s2)
{
    double left1 = p1.x() - s1.width() / 2;
    double left2 = p2.x() - s2.width() / 2;
    return left1 > left2 ? left1 : left2;
}

double minOfRight(Coord const &p1, const Size &s1, Coord const &p2, const Size &s2)
{
    double right1 = p1.x() + s1.width() / 2;
    double right2 = p2.x() + s2.width() / 2;
    return right1 < right2 ? right1 : right2;
}

double minOfTop(Coord const &p1, const Size &s1, Coord const &p2, const Size &s2)
{
    double top1 = p1.y() + s1.height() / 2;
    double top2 = p2.y() + s2.height() / 2;
    return top1 < top2 ? top1 : top2;
}

double maxOfBot(Coord const &p1, const Size &s1, Coord const &p2, const Size &s2)
{
    double bot1 = p1.y() - s1.height() / 2;
    double bot2 = p2.y() - s2.height() / 2;
    return bot1 > bot2 ? bot1 : bot2;
}

tuple<double, double> nodeRectanglesIntersection(Coord const &p1, const Size &s1, Coord const &p2, const Size &s2)
{
    double min_of_right = minOfRight(p1, s1, p2, s2);
    double max_of_left = maxOfLeft(p1, s1, p2, s2);
    double min_of_top = minOfTop(p1, s1, p2, s2);
    double max_of_bot = maxOfBot(p1, s1, p2, s2);

    double intersection_width = min_of_right - max_of_left;
    double intersection_height = min_of_top - max_of_bot;
    return make_tuple(intersection_width, intersection_height);
}

vector<size_t> sort_indexes(const vector<double> &v)
{

    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2)
                { return v[i1] < v[i2]; });

    return idx;
}

vector<size_t> sortNodesByX(vector<double> &X, vector<double> &S, vector<double> &sortedX, vector<double> &sortedS)
{
    int N = X.size() / 2;

    vector<double> xs(N);
    for (int i = 0; i < N; ++i)
        xs[i] = X[i * 2];
    vector<size_t> idx = sort_indexes(xs);
    for (int i = 0; i < N; ++i)
    {
        sortedX[i * 2] = X[2 * idx[i]];
        sortedX[i * 2 + 1] = X[2 * idx[i] + 1];

        sortedS[i * 2] = S[2 * idx[i]];
        sortedS[i * 2 + 1] = S[2 * idx[i] + 1];
    }
    return idx;
}

bool overlapCheck(Coord const &p1, Coord const &p2, const Size &s1, const Size &s2)
{
    auto [i_w, i_h] = nodeRectanglesIntersection(p1, s1, p2, s2);
    if (i_w > 0 && i_h > 0)
        return true;
    return false;
}

bool scanLineOverlapCheck(vector<double> &X, vector<double> &S)
{
    int N = X.size() / 2;
    vector<double> sortedX(2 * N);
    vector<double> sortedS(2 * N);
    sortNodesByX(X, S, sortedX, sortedS);

    Coord p1, p2;
    Size s1, s2;
    double left_j, right_i;
    for (int i = 0; i < N; ++i)
    {
        p1.set(
            sortedX[i * 2],
            sortedX[i * 2 + 1]);
        s1.set(
            sortedS[i * 2],
            sortedS[i * 2 + 1]);

        right_i = p1.x() + s1.width() / 2;
        for (int j = i + 1; j < N; ++j)
        {
            p2.set(sortedX[j * 2], sortedX[j * 2 + 1]);
            s2.set(sortedS[j * 2], sortedS[j * 2 + 1]);

            left_j = p2.x() - s2.width() / 2;
            if (overlapCheck(p1, p2, s1, s2))
                return true;
            else if (left_j > right_i)
                break;
        }
    }
    return false;
}

vector<tuple<int, int>> getAllOverlaps(vector<double> &X, vector<double> &S)
{

    int N = X.size() / 2;

    vector<double> sortedX(X);
    vector<double> sortedS(S);
    vector<size_t> mapping = sortNodesByX(X, S, sortedX, sortedS);

    Coord p1, p2;
    Size s1, s2;
    double left_j, right_i;
    vector<tuple<int, int>> overlaps;
    for (int i = 0; i < N; ++i)
    {
        p1.set(sortedX[i * 2], sortedX[i * 2 + 1]);
        s1.set(sortedS[i * 2], sortedS[i * 2 + 1]);
        right_i = p1.x() + s1.width() / 2;
        for (int j = i + 1; j < N; ++j)
        {
            p2.set(sortedX[j * 2], sortedX[j * 2 + 1]);
            s2.set(sortedS[j * 2], sortedS[j * 2 + 1]);
            left_j = p2.x() - s2.width() / 2;
            if (overlapCheck(p1, p2, s1, s2))
                overlaps.emplace_back(make_tuple(mapping[i], mapping[j]));
            else if (left_j > right_i)
                break;
        }
    }
    return overlaps;
}

double FORBID_delta(Coord const &ci, Coord const &cj, const Size &si, const Size &sj, double intersec_width, double intersec_height)
{
    return sqrt(pow(si.width() / 2 + sj.width() / 2, 2.) + pow(si.height() / 2 + sj.height() / 2, 2.));
}

double SIDE2SIDE_delta(Coord const &ci, Coord const &cj, const Size &si, const Size &sj, double intersec_width, double intersec_height)
{
    double d_eucl = eucl(ci, cj);
    return d_eucl + min(abs(intersec_width), abs(intersec_height));
}

double PRISM_delta(Coord const &ci, Coord const &cj, const Size &si, const Size &sj, double intersec_width, double intersec_height)
{
    double t_ij = max(
        1.,
        min((si.width() / 2 + sj.width() / 2) / abs(ci.x() - cj.x()),
            (si.height() / 2 + sj.height() / 2) / abs(ci.y() - cj.y())));

    return t_ij * eucl(ci, cj);
}
double eucl(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x1 - x2, 2.) + pow(y1 - y2, 2.));
}

double eucl(Coord const &p1, Coord const &p2)
{
    return sqrt(pow(p1.x() - p2.x(), 2.) + pow(p1.y() - p2.y(), 2.));
}

double vecNorm2D(double vec_x, double vec_y)
{
    return sqrt(vec_x * vec_x + vec_y * vec_y);
}

double maxScaleRatio(vector<double> &X, vector<double> &S)
{
    double padding = 1e-4;
    double maxRatio = 1.;
    double optimalDist, actualDist, ratio, unoverlapRatio;
    double actualX, actualY, desiredWidth, desiredHeight, widthRatio, heightRatio;
    auto overlaps = getAllOverlaps(X, S);
    for (unsigned int i = 0; i < overlaps.size(); ++i)
    {
        auto [u, v] = overlaps[i];
        actualDist = eucl(X[u * 2], X[u * 2 + 1], X[v * 2], X[v * 2 + 1]);

        actualX = X[u * 2] - X[v * 2];
        actualY = X[u * 2 + 1] - X[v * 2 + 1];
        desiredWidth = (S[u * 2] + S[v * 2]) / 2 + padding;
        desiredHeight = (S[u * 2 + 1] + S[v * 2 + 1]) / 2 + padding;

        widthRatio = desiredWidth / actualX;
        heightRatio = desiredHeight / actualY;

        unoverlapRatio = min(abs(widthRatio), abs(heightRatio));
        actualX *= unoverlapRatio;
        actualY *= unoverlapRatio;

        optimalDist = vecNorm2D(actualX, actualY);
        ratio = optimalDist / actualDist;

        // ratio = max(
        // 1.,
        // min((S[u*2] / 2 + S[v*2] / 2 + padding) / abs(X[u*2] - X[v*2]),
        //     (S[u*2+1] / 2 + S[v*2+1] / 2 + padding) / abs(X[u*2+1] - X[v*2+1])));

        maxRatio = max(maxRatio, ratio);
    }
    return maxRatio;
}

void scaleLayout(vector<double> &X, double scaleFactor)
{
    int N = X.size() / 2;
    for (int i = 0; i < N; ++i)
    {
        X[i * 2] *= scaleFactor;
        X[i * 2 + 1] *= scaleFactor;
    }
}

bool isCurrentScaleSolvable(vector<double> &X, vector<double> &S)
{
    int N = X.size() / 2;
    double areas_sum = 0;
    double min_x = DBL_MAX;
    double min_y = DBL_MAX;
    double max_x = -DBL_MAX;
    double max_y = -DBL_MAX;

    double left, right, top, bot;
    for (int i = 0; i < N; ++i)
    {
        areas_sum += S[i * 2] * S[i * 2 + 1];
        left = X[i * 2] - S[i * 2] / 2;
        right = X[i * 2] + S[i * 2] / 2;
        top = X[i * 2 + 1] + S[i * 2 + 1] / 2;
        bot = X[i * 2 + 1] - S[i * 2 + 1] / 2;
        if (left < min_x)
            min_x = left;
        if (right > max_x)
            max_x = right;
        if (bot < min_y)
            min_y = bot;
        if (top > max_y)
            max_y = top;
    }
    double bb_area = (max_x - min_x) * (max_y - min_y);

    return bb_area >= areas_sum;
}

string save(vector<double> const &X, vector<double> const &S, string input_path, double elapsed, bool isPrime)
{
    auto N = X.size() / 2;

    string sep = " ";
    string extension = isPrime ? ".forbidp" : ".forbid";
    ofstream f(input_path + extension);
    f << elapsed << endl;
    for (unsigned int i = 0; i < N; ++i)
    {
        f << X[i * 2] << sep << X[i * 2 + 1] << sep << S[i * 2] << sep << S[i * 2 + 1] << endl;
    }
    f.close();
    return input_path + extension;
}

void loadParams(Parametrizer &params, char *argv[])
{
    params.ALPHA = atof(argv[2]);
    params.K = atof(argv[3]);
    params.MINIMUM_MOVEMENT = atof(argv[4]);
    params.MAX_ITER = atoi(argv[5]);
    params.MAX_PASSES = atoi(argv[6]);
    params.SCALE_STEP = atof(argv[7]);
    params.PRIME = atoi(argv[8]);
}

vector<double> initLayout(Layout const &g)
{
    vector<double> X(2 * g.N());
    for (unsigned int i = 0; i < g.N(); i++)
    {
        X[i * 2] = g.pos[i].x();
        X[i * 2 + 1] = g.pos[i].y();
    }
    return X;
}

vector<double> initSizes(Layout const &g)
{
    vector<double> S(2 * g.N());
    for (unsigned int i = 0; i < g.N(); i++)
    {
        S[i * 2] = g.sizes[i].width();
        S[i * 2 + 1] = g.sizes[i].height();
    }
    return S;
}

void copyLayout(vector<double> const &in, vector<double> &out)
{
    auto N = out.size() / 2;
    for (unsigned int i = 0; i < 2 * N; ++i)
        out[i] = in[i];
}

Layout loadOriginalDistance(string input_path)
{
    ifstream f;
    f.open(input_path);
    int N;
    f >> N;

    vector<Coord> pos;
    pos.reserve(N);
    vector<Size> sizes;
    sizes.reserve(N);

    double x, y, w, h;
    for (int i = 0; i < N; i++)
    {
        f >> x >> y >> w >> h;
        pos.emplace_back(x, y);
        sizes.emplace_back(w, h);
    }

    f.close();

    Layout g = Layout(pos, sizes);
    return g;
}

// S_GD2 function, taken from https://github.com/jxz12/s_gd2/blob/master/cpp/s_gd2/layout.cpp
vector<double> schedule(const vector<term> &terms, int t_max, double eps)
{
    double w_min = terms[0].w, w_max = terms[0].w;
    for (unsigned i = 1; i < terms.size(); i++)
    {
        const double &w = terms[i].w;
        if (w < w_min)
            w_min = w;
        if (w > w_max)
            w_max = w;
    }
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;
    double lambda = log(eta_max / eta_min) / ((double)t_max - 1);

    // initialize step sizes
    vector<double> etas;
    etas.reserve(t_max);
    for (int t = 0; t < t_max; t++)
        etas.push_back(eta_max * exp(-lambda * t));

    return etas;
}

void fisheryates_shuffle(vector<term> &terms, rk_state &rstate)
{
    int n = terms.size();
    for (unsigned i = n - 1; i >= 1; i--)
    {
        unsigned j = rk_interval(i, &rstate);
        term temp = terms[i];
        terms[i] = terms[j];
        terms[j] = temp;
    }
}

vector<term> layoutToTerms(vector<double> &X, vector<double> &init_X, vector<double> &S, Parametrizer &params)
{
    int N = X.size() / 2;
    vector<term> terms;
    double d_ij, w_ij; //, d_eucl;
    double xi, xj, yi, yj;
    Coord ci, cj;
    Coord init_ci, init_cj;
    Size si, sj;
    bool overlap;
    for (int i = 0; i < N; i++)
    {
        ci.set(X[i * 2], X[i * 2 + 1]);
        init_ci.set(init_X[i * 2], init_X[i * 2 + 1]);
        si.set(S[i * 2], S[i * 2 + 1]);

        for (int j = i + 1; j < N; j++)
        {
            if (i != j)
            {
                cj.set(X[j * 2], X[j * 2 + 1]);
                init_cj.set(init_X[j * 2], init_X[j * 2 + 1]);
                sj.set(S[j * 2], S[j * 2 + 1]);
                auto [i_w, i_h] = nodeRectanglesIntersection(ci, si, cj, sj);

                if (i_w > 0 && i_h > 0) // there is overlap
                {
                    d_ij = (*params.distance_fn)(init_ci, init_cj, si, sj, i_w, i_h);
                    w_ij = pow(d_ij, params.K * params.ALPHA);
                    overlap = true;
                }
                else
                {
                    d_ij = eucl(ci, cj);
                    w_ij = pow(d_ij, params.ALPHA);
                    overlap = false;
                }
                terms.push_back(term(i, j, d_ij, w_ij, overlap));
            }
        }
    }
    return terms;
}

// S_GD2 optim algorithm, adapted from https://github.com/jxz12/s_gd2/blob/master/cpp/s_gd2/layout.cpp
void OPTIMIZATION_PASS(vector<double> &X, vector<term> &terms, vector<double> &init_X, vector<double> &S, const vector<double> &etas, Parametrizer &params)
{
    rk_state rstate;
    rk_seed(params.seed, &rstate);
    double mvt_sum;
    unsigned int i_eta;
    for (i_eta = 0; i_eta < etas.size(); i_eta++)
    {
        const double eta = etas[i_eta];

        unsigned n_terms = terms.size();
        if (n_terms == 0)
            return;
        fisheryates_shuffle(terms, rstate);

        mvt_sum = 0;
        for (unsigned i_term = 0; i_term < n_terms; i_term++)
        {
            const term &t = terms[i_term];
            const int &i = t.i, &j = t.j;
            const double &w_ij = t.w;
            const double &d_ij = t.d;
            if (true || t.o)
            {
                double mu = eta * w_ij;
                if (mu > 1)
                    mu = 1;
                double dx = X[i * 2] - X[j * 2];
                double dy = X[i * 2 + 1] - X[j * 2 + 1];
                double mag = sqrt(dx * dx + dy * dy);

                double r = 0;
                if (mag != 0)
                    r = (mu * (mag - d_ij)) / (2 * mag);
                double r_x = r * dx;
                double r_y = r * dy;
                mvt_sum += abs(r_x) + abs(r_y);
                X[i * 2] -= r_x;
                X[i * 2 + 1] -= r_y;
                X[j * 2] += r_x;
                X[j * 2 + 1] += r_y;
            }
        }
        if (mvt_sum < params.MINIMUM_MOVEMENT)
        {
            return;
        }
        terms = layoutToTerms(X, init_X, S, params);
    }
    return;
}

void passInOptim(vector<double> &X, vector<double> &initX, vector<double> &S, Parametrizer &params)
{
    vector<term> orig_terms = layoutToTerms(X, initX, S, params);
    vector<double> etas = schedule(orig_terms, params.MAX_ITER, params.eps);
    OPTIMIZATION_PASS(X, orig_terms, initX, S, etas, params);
}

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        cerr << "Wrong parameters" << endl;
        return EXIT_FAILURE;
    }
    string input_layout_path = argv[1];
    Layout g = loadOriginalDistance(input_layout_path);

    Parametrizer params;
    loadParams(params, argv);

    auto init_X = initLayout(g);
    vector<double> scaled_init_X(g.N() * 2);
    auto S = initSizes(g);
    vector<double> X(g.N() * 2);
    copyLayout(init_X, scaled_init_X);
    copyLayout(init_X, X);

    int n_passes = 0;
    double scaleFactor;

    double maxRatio = maxScaleRatio(X, S);
    double upperScale = maxRatio;
    double lowerScale = 1.;
    double oldScale = 1.;
    double curScale = 1.;

    auto start = std::chrono::steady_clock::now();
    bool stop = !scanLineOverlapCheck(X, S); // do not enter if there is no overlap

    // if possible, try to solve the problem in current scale
    if (!stop && isCurrentScaleSolvable(X, S))
    {
        passInOptim(X, init_X, S, params);
        stop = !scanLineOverlapCheck(X, S);
    }

    while (!stop)
    {
        curScale = (upperScale + lowerScale) / 2;
        scaleFactor = curScale / oldScale;
        oldScale = curScale;
        scaleLayout(scaled_init_X, scaleFactor);
        if (params.PRIME)
            copyLayout(scaled_init_X, X);
        else
            scaleLayout(X, scaleFactor);

        passInOptim(X, scaled_init_X, S, params);

        if (scanLineOverlapCheck(X, S))
            lowerScale = curScale;
        else // no overlap
        {
            if (upperScale - lowerScale < params.SCALE_STEP)
                stop = true;
            else
                upperScale = curScale;
        }

        ++n_passes;
        if (n_passes >= params.MAX_PASSES)
            break;
    }
    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    string savepath = save(X, S, input_layout_path, elapsed, params.PRIME);

    cout << "DONE; saved to " << savepath << endl;
    return EXIT_SUCCESS;
}

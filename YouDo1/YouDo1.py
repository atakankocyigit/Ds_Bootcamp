from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import plotly.graph_objects as go


def weDo_reg(x, y, group, p=1.0, verbose=False):
    beta = np.random.random(2)
    gamma = dict((k, np.random.random(2)) for k in range(6))

    if verbose:
        st.write(beta)
        st.write(gamma)
        st.write(x)

    alpha = 0.000001
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _k, _x, _y in zip(group, x, y):
            y_pred = p * (beta[0] + beta[1] * _x) + (1 - p) * (gamma[_k][0] + gamma[_k][1] * _x)

            g_b0 = -2 * p * (_y - y_pred)
            g_b1 = -2 * p * ((_y - y_pred) * _x)

            g_g0 = -2 * (1 - p) * (_y - y_pred)
            g_g1 = -2 * (1 - p) * ((_y - y_pred) * _x)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            gamma[_k][0] = gamma[_k][0] - alpha * g_g0
            gamma[_k][1] = gamma[_k][1] - alpha * g_g1

            err += (_y - y_pred) ** 2

        print(f"WeDo Regression: {it} - Beta: {beta}, Error: {err / len(y)}")
        my_bar.progress(it / n_max_iter)
    return beta


def make_reg(verbose: bool = False, n: int = 100):
    rng = np.random.RandomState(0)
    st.markdown(r"Convexity Visualization for $\beta_0$ and $\beta_1$ in the sample dataset")
    x, y = make_regression(n, 1, random_state=rng, noise=5.)

    df = pd.DataFrame(dict(x=x[:, 0], y=y))

    if verbose:
        fig = px.scatter(df, x="x", y="y", trendline="ols")

        st.plotly_chart(fig, use_container_width=True)

    return x, y


def convexity(verbose, threshold: int = 100):
    samples = 150
    x, y = make_reg(verbose, samples)
    m, n = 100, 100
    b0 = np.linspace(-40, 70, m)
    b1 = np.linspace(-40, 70., n)

    my_bar = st.progress(0.)
    loss = np.empty((m, n))
    threshold = 900
    for i, _b0 in enumerate(b0):
        for j, _b1 in enumerate(b1):
            error = (y.reshape(y.shape[0], 1) - (_b1 * x + _b0)) ** 2
            y_pred_thresh = np.array([err if err <= threshold else threshold + err / 10 ** 5 for err in error])

            loss[i][j] = y_pred_thresh.sum()

        my_bar.progress(i / m)
    # FIX: Axis naming

    fig = go.Figure(data=go.Contour(
        z=loss,
        x=b0,
        y=b1
    ))

    # fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def ls_l2(x, y, threshold, lam, alpha, n_max_iter=200) -> np.ndarray:
    beta = np.random.random(2)

    my_bar = st.progress(0.)
    for it in range(n_max_iter):
        error, g_b0, g_b1 = 0, 0, 0
        beta_prev = beta.copy()
        for _x, _y in zip(x, y):
            y_pred: np.ndarray = beta[0] + beta[1] * _x

            g_b0 += -2 * (_y - y_pred) + 2 * lam * beta[0] if (_y - y_pred) ** 2 <= threshold else -2 * (
                    _y - y_pred) / 10 ** 5 + 2 * lam * beta[0]
            g_b1 += -2 * ((_y - y_pred) * _x) + 2 * lam * beta[1] if (_y - y_pred) ** 2 <= threshold else -2 * (
                    (_y - y_pred) * _x) / 10 ** 5 + 2 * lam * beta[1]

            error += (_y - y_pred) ** 2 if (_y - y_pred) ** 2 <= threshold else threshold + (
                    (_y - y_pred) ** 2) / 10 ** 5

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        print(f"L2 Regularization ({it}) beta: {beta}, gradient: {g_b0} {g_b1}, Error: {error / len(y)}")

        my_bar.progress(it / n_max_iter)
        if np.linalg.norm(beta - beta_prev) < 0.00001:
            my_bar.progress(n_max_iter / n_max_iter)
            print(f"I do early stopping at iteration {it}")
            break

    return beta


def ls(x, y, threshold, alpha, n_max_iter=200):
    beta = np.random.random(2)

    my_bar = st.progress(0.)
    for it in range(n_max_iter):
        error, g_b0, g_b1 = 0, 0, 0
        beta_prev = beta.copy()
        for _x, _y in zip(x, y):
            y_pred: np.ndarray = beta[0] + beta[1] * _x

            g_b0 += -2 * (_y - y_pred) if (_y - y_pred) ** 2 <= threshold else -2 * (_y - y_pred) / 10 ** 5
            g_b1 += -2 * ((_y - y_pred) * _x) if (_y - y_pred) ** 2 <= threshold else -2 * (
                    (_y - y_pred) * _x) / 10 ** 5

            error += (_y - y_pred) ** 2 if (_y - y_pred) ** 2 <= threshold else threshold + (
                    (_y - y_pred) ** 2) / 10 ** 5

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        print(f"Non-Regularized ({it}) - Beta: {beta}, gradient: {g_b0} {g_b1}, Error: {error / len(y)}")

        my_bar.progress(it / n_max_iter)
        if np.linalg.norm(beta - beta_prev) < 0.00001:
            print(f"I do early stopping at iteration {it}")
            break

    return beta


def main(verbose: bool = True):
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))
    if verbose:
        st.header("Dataset")
        st.dataframe(df)
        st.write(df.describe())

    st.header("1. Loss Function")

    st.markdown("A condition is made in the loss function for the loss function to be approximately equal to the "
                "threshold if it exceeds a certain threshold value.")

    st.markdown(r"Our New Loss function can be written as: ")

    st.latex(r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - ("
             r"\beta_0 + \beta_1 x_i))^2 }")
    st.latex(
        r"else: (y_i - (\beta_0 + \beta_1 x_i))^2 > Threshold, L(\beta_0, \beta_1) = \sum_{i=1}^{N}{Threshold + (y_i "
        r"- (\beta_0 + \beta_1 x_i))^2 } / 10^5")

    st.header("2. Loss Function Convexity Visualization")
    convexity(verbose)

    st.header("2.5 Derivatives of Loss Function")

    st.markdown(
        r"Partial derivatives "
        r"For $\beta_0$: ")

    st.latex(
        r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, \frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i )}")

    st.latex(
        r"else: \frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i ) / 10^5}")

    st.markdown(r"For $\beta_1$: ")
    st.latex(
        r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, \frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i}")

    st.latex(
        r"else: \frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i / 10^5}")

    threshold = st.slider("Threshold According to Squared Error", 0., 50., value=4.)

    st.header("3. Optimum beta values for Non-Regularized Model: ")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["MedInc"]
                   , y=np.full(len(df["MedInc"]), fill_value=np.mean(y)), mode='lines', name='bias only'))

    alphal2, alphals = 0.000001, 0.000001

    betalr = ls(X_train["MedInc"], y_train, threshold, alphals)
    st.latex(fr"\beta_0={betalr[0]:.4f}, \beta_1={betalr[1]:.4f}")

    st.header("4. L2 Regularization: ")

    lam1 = st.slider("Regularization Multiplier for L2 (lambda)", 0.001, 10., value=4.2)
    betal2 = ls_l2(X_train["MedInc"], y_train, threshold, lam1, alphal2)
    st.latex(
        r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - ("
        r"\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2), \lambda > 0")

    st.latex(
        r"else: L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 / 10^5} + Threshold + \lambda ("
        r"\beta_0^2 + \beta_1^2), \lambda > 0")

    st.markdown(r"For $\beta_0$: ")
    st.latex(
        r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, \frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i )} + 2 \lambda \beta_0")

    st.latex(
        r"else: \frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i ) / 10^5} + 2 \lambda \beta_0")

    st.markdown(r"For $\beta_1$: ")
    st.latex(
        r"if: (y_i - (\beta_0 + \beta_1 x_i))^2 <= Threshold, \frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{"
        r"i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i} + 2 \lambda \beta_1")

    st.latex(
        r"else: \frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i / 10^5} "
        r"+ 2 \lambda \beta_1")

    st.write("Optimum beta values for L2 Regularized Model: ")
    st.latex(fr"\beta_0={betal2[0]:.4f}, \beta_1={betal2[1]:.4f}")

    fig.add_trace(
        go.Scatter(x=df["MedInc"], y=betalr[0] + betalr[1] * df["MedInc"], mode='lines',
                   name='regression'))

    fig.add_trace(
        go.Scatter(x=df["MedInc"], y=betal2[0] + betal2[1] * df["MedInc"], mode='lines',
                   name='regression + L2 (Lambda = 100)'))

    st.header("5. YouDo vs WeDo Regression Models:")
    beta_weDo = weDo_reg(X_train["MedInc"], y_train, (X['HouseAge'].values / 10).astype(np.int))

    st.write("Optimum beta values for We Do Regression Model: ")
    st.latex(fr"\beta_0={beta_weDo[0]:.4f}, \beta_1={beta_weDo[1]:.4f}")

    fig.add_trace(
        go.Scatter(x=df["MedInc"], y=beta_weDo[0] + beta_weDo[1] * df["MedInc"], mode='lines',
                   name='We Do regression'))

    fig.add_trace(go.Scatter(x=df["MedInc"], y=y, mode='markers', name='data points'))
    st.plotly_chart(fig, use_container_width=True)

    st.write(fr"**Non-Regularized MSE:** {((y - (betalr[0] + betalr[1] * df['MedInc'])) ** 2).mean():.4f}")
    st.write(fr"**L2 Regularization MSE:** {((y - (betal2[0] + betal2[1] * df['MedInc'])) ** 2).mean():.4f}")
    st.write(fr"**WeDo Linear Regression MSE:** {((y - (beta_weDo[0] + beta_weDo[1] * df['MedInc'])) ** 2).mean():.4f}")
    st.write(fr"**Mean MSE:** {((y - y.mean()) ** 2).mean():.4f}")


if __name__ == '__main__':
    main(verbose=st.sidebar.checkbox("Verbose", value=False))

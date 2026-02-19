import styles from './CommonOverrideBanner.module.scss';

const OverrideBanner = () => {
  if (!sessionStorage.getItem(`df_date_override`) || process.env.NODE_ENV !== `development`) {
    return null;
  }
  const overrideDateISO = sessionStorage.getItem(`df_date_override`);
  let formattedDate = ``;
  if (overrideDateISO) {
    const date = new Date(overrideDateISO);
    formattedDate = `${(date.getMonth() + 1).toString().padStart(2, `0`)}/${date
      .getDate()
      .toString()
      .padStart(2, `0`)}/${date.getFullYear()}`;
  }
  return <div className={styles.banner}>You have set the date override to {formattedDate}</div>;
};

export default OverrideBanner;
